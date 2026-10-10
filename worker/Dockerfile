FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/sat3dgen
COPY requirements.txt /tmp/sat-requirements.txt
COPY worker/requirements.txt /tmp/worker-requirements.txt
RUN pip install --no-cache-dir -r /tmp/sat-requirements.txt -r /tmp/worker-requirements.txt

COPY . /opt/sat3dgen
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    SAT3DGEN_WORK=/cache/jobs \
    WORKER_HOST=0.0.0.0 \
    WORKER_PORT=8088
EXPOSE 8088
CMD ["python", "-m", "worker.server"]
