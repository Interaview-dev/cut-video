import os
import shutil
import subprocess
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scenedetect import ContentDetector, detect

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")  # ex: https://mon-service.onrender.com
DATA = Path(os.environ.get("DATA_DIR", "/tmp/data"))
DATA.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/files", StaticFiles(directory=DATA), name="files")


class SplitRequest(BaseModel):
    video_url: str
    threshold: float = 27.0   # plus bas = plus sensible
    min_scene_len: int = 15   # en images


def check_key(key: str):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")


@app.post("/split")
def split(req: SplitRequest, x_api_key: str = Header(default="")):
    check_key(x_api_key)
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA / job_id
    job_dir.mkdir()
    src = job_dir / "source.mp4"

    # 1. Téléchargement
    with requests.get(req.video_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(src, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

    # 2. Détection des plans
    scenes = detect(
        str(src),
        ContentDetector(threshold=req.threshold, min_scene_len=req.min_scene_len),
    )
    if not scenes:  # aucun changement détecté : la vidéo entière est un seul plan
        raise HTTPException(status_code=422, detail="Aucun plan détecté")

    # 3. Découpe (ré-encodage pour des coupes précises à l'image)
    results = []
    for i, (start, end) in enumerate(scenes, start=1):
        name = f"plan_{i:03d}.mp4"
        s, e = start.get_seconds(), end.get_seconds()
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{s:.3f}", "-i", str(src),
             "-t", f"{e - s:.3f}", "-c:v", "libx264", "-preset", "veryfast",
             "-c:a", "aac", str(job_dir / name)],
            check=True, capture_output=True,
        )
        results.append({
            "plan": i,
            "debut": start.get_timecode(),
            "fin": end.get_timecode(),
            "duree_s": round(e - s, 3),
            "url": f"{BASE_URL}/files/{job_id}/{name}",
        })

    src.unlink()  # on garde seulement les plans
    return {"job_id": job_id, "nb_plans": len(results), "plans": results}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, x_api_key: str = Header(default="")):
    check_key(x_api_key)
    shutil.rmtree(DATA / job_id, ignore_errors=True)
    return {"deleted": job_id}
