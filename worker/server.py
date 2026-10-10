"""One Sat3DGen mesh at a time.

A finished mesh is a generated street-level scene from one satellite image.
It is not a photograph, a parcel survey, or a live satellite view.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("SAT3DGEN_WORK", ROOT / "worker_jobs")).resolve()
MODEL = os.environ.get("SAT3DGEN_MODEL", "qian43/Sat3DGen")
TOKEN = os.environ.get("WORKER_TOKEN", "")
MAX_BYTES = int(os.environ.get("SAT3DGEN_MAX_BYTES", str(8 * 1024 * 1024)))
TIMEOUT = int(os.environ.get("SAT3DGEN_TIMEOUT", "1200"))
HOST = os.environ.get("WORKER_HOST", "127.0.0.1")
PORT = int(os.environ.get("WORKER_PORT", "8088"))
NOTICE = "Generated scene. Not a photograph and not a survey of the parcel."

LOCK = threading.Lock()
RUNNING: str | None = None
ALLOWED = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}

app = FastAPI(title="Sat3DGen worker", version="1")


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def require_token(authorization: str | None) -> None:
    if not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Missing or wrong worker token.")


def job_dir(job_id: str) -> Path:
    if not job_id or any(part in job_id for part in ("/", "\\", "..")) or job_id.startswith("."):
        raise HTTPException(status_code=404, detail="Unknown job.")
    path = (WORK / job_id).resolve()
    if path.parent != WORK or not path.is_dir():
        raise HTTPException(status_code=404, detail="Unknown job.")
    return path


def read_status(path: Path) -> dict:
    raw = path / "status.json"
    if not raw.is_file():
        raise HTTPException(status_code=404, detail="Unknown job.")
    return json.loads(raw.read_text())


def write_status(path: Path, status: dict) -> None:
    (path / "status.json").write_text(json.dumps(status))


def find_mesh(path: Path) -> Path | None:
    matches = sorted(path.rglob("mesh_ori.obj"))
    return matches[0] if matches else None


def run_job(path: Path, image: Path, job_id: str) -> None:
    global RUNNING
    status = read_status(path)
    status["state"] = "running"
    write_status(path, status)
    command = [
        "python",
        "inference/single_image_inference.py",
        "--sat_img_path",
        str(image),
        "--model_path",
        MODEL,
        "--work_dir",
        str(path / "out"),
        "--only_save_mesh",
        "true",
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
        (path / "log.txt").write_text(((proc.stdout or "") + (proc.stderr or ""))[-8000:])
        mesh = find_mesh(path / "out")
        if proc.returncode != 0 or mesh is None:
            status["state"] = "failed"
            status["error"] = "Inference did not write a mesh."
        else:
            status["state"] = "done"
            status["mesh"] = str(mesh.relative_to(path))
            status["notice"] = NOTICE
    except subprocess.TimeoutExpired:
        status["state"] = "failed"
        status["error"] = "Inference timed out."
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = type(exc).__name__
    write_status(path, status)
    with LOCK:
        if RUNNING == job_id:
            RUNNING = None


@app.get("/health")
def health():
    return {"ok": True, "cuda": cuda_available(), "busy": RUNNING is not None, "model": MODEL, "notice": NOTICE}


@app.post("/v1/jobs")
async def create_job(authorization: str | None = Header(default=None), image: UploadFile = File(...)):
    global RUNNING
    require_token(authorization)
    if not cuda_available():
        raise HTTPException(status_code=503, detail="No CUDA GPU. This worker does not run the model on CPU.")
    suffix = ALLOWED.get(image.content_type or "")
    if not suffix:
        raise HTTPException(status_code=415, detail="Send a PNG, JPEG, or WebP satellite image.")
    data = await image.read()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is empty or larger than 8 MB.")
    with LOCK:
        if RUNNING is not None:
            raise HTTPException(status_code=429, detail="A job is already running.")
        job_id = uuid.uuid4().hex
        RUNNING = job_id
    path = WORK / job_id
    try:
        path.mkdir(parents=True, exist_ok=False)
        image_path = path / f"input{suffix}"
        image_path.write_bytes(data)
        status = {"id": job_id, "state": "queued", "notice": NOTICE}
        write_status(path, status)
    except Exception:
        with LOCK:
            if RUNNING == job_id:
                RUNNING = None
        raise
    threading.Thread(target=run_job, args=(path, image_path, job_id), daemon=True).start()
    return status


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(default=None)):
    require_token(authorization)
    return read_status(job_dir(job_id))


@app.get("/v1/jobs/{job_id}/mesh")
def get_mesh(job_id: str, authorization: str | None = Header(default=None)):
    require_token(authorization)
    path = job_dir(job_id)
    status = read_status(path)
    if status.get("state") != "done" or not status.get("mesh"):
        raise HTTPException(status_code=409, detail="Mesh is not ready.")
    mesh = (path / status["mesh"]).resolve()
    if not mesh.is_relative_to(path) or not mesh.is_file():
        raise HTTPException(status_code=404, detail="Mesh file is missing.")
    return FileResponse(mesh, media_type="model/obj", filename="mesh_ori.obj")


def main():
    import uvicorn

    WORK.mkdir(parents=True, exist_ok=True)
    uvicorn.run("worker.server:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
