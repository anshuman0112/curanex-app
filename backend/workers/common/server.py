"""Shared model-worker server (BACKEND_PLAN §3.1). Phase B0: no models yet.
Phase B2 adds the ModelAdapter registry, POST /predict/{model_id}, lazy loading, LRU and timeouts."""
import os

from fastapi import FastAPI

WORKER = os.getenv("WORKER_NAME", "worker")

app = FastAPI(title=f"Curanex {WORKER}")


@app.get("/health")
def health():
    return {"ok": True, "worker": WORKER}


@app.get("/models")
def models():
    return []
