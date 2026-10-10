# Sat3DGen GPU worker

Runs one mesh job at a time by calling `inference/single_image_inference.py --only_save_mesh true`.

The mesh is a generated street-level scene. It is not a photograph of the property and it is not a parcel survey.

This machine that committed the worker has no CUDA device. The server starts anyway. `POST /v1/jobs` returns 503 until `torch.cuda.is_available()` is true. The first real job downloads about 1.5 GB of weights from `qian43/Sat3DGen`.

## Run

From the repository root, on a machine with an NVIDIA GPU:

```bash
pip install -r requirements.txt -r worker/requirements.txt
export WORKER_TOKEN="$(openssl rand -hex 24)"
export WORKER_HOST=0.0.0.0
python -m worker.server
```

Docker, from the repository root:

```bash
docker build -f worker/Dockerfile -t sat3dgen-worker .
docker run --gpus all -p 8088:8088 \
  -e WORKER_TOKEN="$WORKER_TOKEN" \
  -v sat3dgen-cache:/cache \
  sat3dgen-worker
```

Keep the cache volume. It holds the weights and finished jobs.

## Calls

`GET /health` reports `cuda` and whether a job is running.

```bash
curl -F "image=@tile.png;type=image/png" \
  -H "Authorization: Bearer $WORKER_TOKEN" \
  http://127.0.0.1:8088/v1/jobs
```

Poll `GET /v1/jobs/{id}`. When `state` is `done`, download `GET /v1/jobs/{id}/mesh`.

Set `WORKER_TOKEN` before opening the port. Without it, anyone who can reach the process can start a job.
