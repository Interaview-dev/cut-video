import logging
import os
import shutil
import subprocess
import threading
import traceback
import uuid
import zipfile
from pathlib import Path

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scenedetect import ContentDetector, detect

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scene-splitter")

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")  # ex: https://mon-service.up.railway.app
DATA = Path(os.environ.get("DATA_DIR", "/tmp/data"))
DATA.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/files", StaticFiles(directory=DATA), name="files")


class SplitRequest(BaseModel):
    video_url: str
    threshold: float = 27.0   # plus bas = plus sensible
    min_scene_len: int = 15   # en images
    reencode: bool = False    # False = découpe sans ré-encodage (rapide, léger, coupes sur images clés)
    run_id: str = ""          # identifiant fourni par l'appelant, permet d'interrompre avec /cancel/{run_id}


# Traitements en cours : run_id -> {"cancel": Event, "proc": processus FFmpeg en cours}
JOBS = {}


def check_cancelled(job):
    if job and job["cancel"].is_set():
        raise HTTPException(status_code=409, detail="Découpage interrompu par l'utilisateur.")


def check_key(key: str):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide (en-tête x-api-key)")


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/split")
def split(req: SplitRequest, x_api_key: str = Header(default="")):
    check_key(x_api_key)
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA / job_id
    job_dir.mkdir()
    src = job_dir / "source.mp4"

    job = None
    if req.run_id:
        job = {"cancel": threading.Event(), "proc": None}
        JOBS[req.run_id] = job

    try:
        # 1. Téléchargement
        try:
            with requests.get(req.video_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                ctype = r.headers.get("content-type", "").lower()
                if "text/html" in ctype:
                    raise HTTPException(
                        status_code=400,
                        detail="L'URL renvoie une page web, pas une vidéo. "
                               "Utilise un lien direct vers le fichier (ex. .mp4).",
                    )
                with open(src, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        check_cancelled(job)
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Téléchargement impossible : {type(e).__name__}: {e}",
            )

        if not src.exists() or src.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Le fichier téléchargé est vide.")

        check_cancelled(job)

        # 2. Détection des plans (ne peut pas être coupée en plein calcul : l'arrêt a lieu juste après)
        try:
            scenes = detect(
                str(src),
                ContentDetector(threshold=req.threshold, min_scene_len=req.min_scene_len),
            )
        except Exception as e:
            log.error(traceback.format_exc())
            raise HTTPException(
                status_code=422,
                detail=f"Impossible d'analyser la vidéo (format non lisible ?) : "
                       f"{type(e).__name__}: {e}",
            )

        check_cancelled(job)

        if not scenes:
            raise HTTPException(
                status_code=422,
                detail="Aucun changement de plan détecté. Essaie de baisser 'threshold' "
                       "(ex. 20), ou la vidéo ne contient qu'un seul plan.",
            )

        # 3. Découpe (ré-encodage pour des coupes précises à l'image)
        results = []
        for i, (start, end) in enumerate(scenes, start=1):
            check_cancelled(job)
            name = f"plan_{i:03d}.mp4"
            s, e = start.get_seconds(), end.get_seconds()
            try:
                if req.reencode:
                    # coupes précises, mais plus lourd (1 thread pour limiter la mémoire)
                    codec = ["-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
                             "-c:a", "aac"]
                else:
                    codec = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
                proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{s:.3f}", "-i", str(src), "-t", f"{e - s:.3f}",
                     *codec, str(job_dir / name)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                if job:
                    job["proc"] = proc
                _, stderr = proc.communicate()
                if job:
                    job["proc"] = None
                check_cancelled(job)
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, proc.args, stderr=stderr)
            except FileNotFoundError:
                raise HTTPException(status_code=500, detail="FFmpeg n'est pas installé dans le conteneur.")
            except subprocess.CalledProcessError as err:
                log.error(err.stderr)
                raise HTTPException(
                    status_code=500,
                    detail=f"FFmpeg a échoué sur le plan {i} (code {err.returncode}"
                           f"{', processus tué : mémoire insuffisante ?' if err.returncode in (-9, 137) else ''}) : "
                           f"{(err.stderr or '')[-500:]}",
                )
            results.append({
                "plan": i,
                "debut": start.get_timecode(),
                "fin": end.get_timecode(),
                "duree_s": round(e - s, 3),
                "url": f"{BASE_URL}/files/{job_id}/{name}",
            })

        src.unlink(missing_ok=True)  # on garde seulement les plans

        # 4. Archive ZIP de tous les plans (pas de compression : les vidéos le sont déjà)
        zip_path = job_dir / "plans.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for mp4 in sorted(job_dir.glob("plan_*.mp4")):
                z.write(mp4, arcname=mp4.name)

        return {
            "job_id": job_id,
            "nb_plans": len(results),
            "zip_url": f"{BASE_URL}/files/{job_id}/plans.zip",
            "plans": results,
        }

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Erreur inattendue : {type(e).__name__}: {e}")
    finally:
        if req.run_id:
            JOBS.pop(req.run_id, None)


@app.post("/cancel/{run_id}")
def cancel(run_id: str, x_api_key: str = Header(default="")):
    check_key(x_api_key)
    job = JOBS.get(run_id)
    if not job:
        return {"cancelled": False, "detail": "Aucun traitement en cours pour ce run_id."}
    job["cancel"].set()
    proc = job.get("proc")
    if proc and proc.poll() is None:
        proc.kill()
    return {"cancelled": True}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, x_api_key: str = Header(default="")):
    check_key(x_api_key)
    shutil.rmtree(DATA / job_id, ignore_errors=True)
    return {"deleted": job_id}
