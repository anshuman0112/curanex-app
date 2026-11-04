# worker-tf: Keras .h5 / .keras / .tflite. Build context: backend/
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 WORKER_NAME=worker-tf WORKER_TYPE=tf
WORKDIR /app
COPY requirements/worker-tf.txt requirements/
RUN pip install -r requirements/worker-tf.txt
COPY workers/__init__.py workers/__init__.py
COPY workers/common workers/common
COPY workers/tf workers/tf
RUN useradd --create-home curanex
USER curanex
EXPOSE 8000
CMD ["uvicorn", "workers.common.server:app", "--host", "0.0.0.0", "--port", "8000"]
