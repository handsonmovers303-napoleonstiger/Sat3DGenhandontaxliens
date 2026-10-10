# Sat3DGen worker

Runs one mesh job at a time by calling `inference/single_image_inference.py --only_save_mesh true`.

The mesh is a generated street-level scene. It is not a photograph of the property and it is not a parcel survey.

`SAT3DGEN_DEVICE` defaults to `auto`. A CUDA GPU is used when this process can see one. Otherwise the same inference path runs on CPU, with `CUDA_VISIBLE_DEVICES` cleared, and the job is labeled `device: cpu`. CPU is slow. The default CPU limit is 6 hours (`SAT3DGEN_CPU_TIMEOUT`). Set `SAT3DGEN_DEVICE=cuda` to refuse the job instead of falling back.

The first real job downloads about 1.5 GB of weights from `qian43/Sat3DGen`.

## Run

From the repository root:

```bash
pip install -r requirements.txt -r worker/requirements.txt
export WORKER_TOKEN="$(openssl rand -hex 24)"
export WORKER_HOST=0.0.0.0
python -m worker.server
```

GPU Docker, from the repository root:

```bash
docker build -f worker/Dockerfile -t sat3dgen-worker .
docker run --gpus all -p 8088:8088 \
  -e WORKER_TOKEN="$WORKER_TOKEN" \
  -v sat3dgen-cache:/cache \
  sat3dgen-worker
```

CPU-only Docker drops `--gpus all`. The image is still the CUDA image; PyTorch uses the CPU when no GPU is visible.

Keep the cache volume. It holds the weights and finished jobs.

## Calls

`GET /health` reports `cuda`, the device the next job will use, and whether a job is running.

```bash
curl -F "image=@tile.png;type=image/png" \
  -H "Authorization: Bearer $WORKER_TOKEN" \
  http://127.0.0.1:8088/v1/jobs
```

Poll `GET /v1/jobs/{id}`. A CPU job includes `device: cpu`. When `state` is `done`, download `GET /v1/jobs/{id}/mesh`.

Set `WORKER_TOKEN` before opening the port. Without it, anyone who can reach the process can start a job.