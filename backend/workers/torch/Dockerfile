# worker-torch: torchxrayvision / timm / ultralytics. Build context: backend/
# CPU wheels for the demo; on a GPU machine drop the extra index to get CUDA builds.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 WORKER_NAME=worker-torch WORKER_TYPE=torch
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/worker-torch.txt requirements/
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/worker-torch.txt
COPY workers/__init__.py workers/__init__.py
COPY workers/common workers/common
COPY workers/torch workers/torch
RUN useradd --create-home curanex
USER curanex
EXPOSE 8000
CMD ["uvicorn", "workers.common.server:app", "--host", "0.0.0.0", "--port", "8000"]
