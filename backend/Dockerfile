# API service. Build context: backend/
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
WORKDIR /app
COPY requirements/api.txt requirements/dev.txt requirements/
# CPU-only torch (pulled in by sentence-transformers) keeps the image small
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/api.txt -r requirements/dev.txt
COPY main.py pyproject.toml ./
COPY api api
COPY registry registry
COPY tests tests
RUN useradd --create-home curanex && mkdir -p /data/uploads && chown curanex /data/uploads
USER curanex
ENV UPLOADS_DIR=/data/uploads
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
