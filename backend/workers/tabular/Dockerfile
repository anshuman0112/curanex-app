# worker-tabular: sklearn / xgboost / lightgbm / ONNX. Build context: backend/
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 WORKER_NAME=worker-tabular WORKER_TYPE=tabular
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/worker-tabular.txt requirements/
RUN pip install -r requirements/worker-tabular.txt
COPY workers/__init__.py workers/__init__.py
COPY workers/common workers/common
COPY workers/tabular workers/tabular
RUN useradd --create-home curanex
USER curanex
EXPOSE 8000
CMD ["uvicorn", "workers.common.server:app", "--host", "0.0.0.0", "--port", "8000"]
