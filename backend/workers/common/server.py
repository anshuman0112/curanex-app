"""Shared model-worker server (BACKEND_PLAN §3.1).

Each worker image sets WORKER_TYPE (tabular / torch / tf). Adapters are the modules in
workers/<type>/adapters/ that define `Adapter(ModelAdapter)`. Models load lazily on first use,
at most MAX_LOADED stay in memory (LRU), and each prediction has a timeout."""
import importlib
import os
import pkgutil
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from workers.common.adapter import ModelAdapter

WORKER_TYPE = os.getenv("WORKER_TYPE", "tabular")
WORKER = os.getenv("WORKER_NAME", f"worker-{WORKER_TYPE}")
MAX_LOADED = int(os.getenv("MAX_LOADED_MODELS", "5"))
TIMEOUT_S = float(os.getenv("PREDICT_TIMEOUT_S", "30"))


def discover(worker_type: str) -> dict[str, type[ModelAdapter]]:
    pkg = importlib.import_module(f"workers.{worker_type}.adapters")
    found = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        cls = getattr(importlib.import_module(f"{pkg.__name__}.{info.name}"), "Adapter", None)
        if cls and cls.id:
            found[cls.id] = cls
    return found


class Registry:
    def __init__(self, adapters: dict[str, type[ModelAdapter]], max_loaded: int = MAX_LOADED):
        self.adapters = adapters
        self.max_loaded = max_loaded
        self.loaded: OrderedDict[str, ModelAdapter] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, model_id: str) -> ModelAdapter:
        with self.lock:
            if model_id in self.loaded:
                self.loaded.move_to_end(model_id)
                return self.loaded[model_id]
            adapter = self.adapters[model_id]()
            adapter.load()
            self.loaded[model_id] = adapter
            if len(self.loaded) > self.max_loaded:
                self.loaded.popitem(last=False)
            return adapter


class PredictRequest(BaseModel):
    inputs: dict = {}


def create_app(worker_type: str = WORKER_TYPE) -> FastAPI:
    registry = Registry(discover(worker_type))
    pool = ThreadPoolExecutor(max_workers=4)
    # Load models in the background at startup so the first user doesn't wait for it
    threading.Thread(target=lambda: [registry.get(i) for i in list(registry.adapters)[:MAX_LOADED]], daemon=True).start()
    app = FastAPI(title=f"Curanex {WORKER}")

    @app.get("/health")
    def health():
        return {"ok": True, "worker": WORKER, "models": len(registry.adapters), "loaded": list(registry.loaded)}

    @app.get("/models")
    def models():
        return sorted(registry.adapters)

    @app.post("/predict/{model_id}")
    def predict(model_id: str, req: PredictRequest):
        if model_id not in registry.adapters:
            raise HTTPException(404, f"Unknown model {model_id}")
        adapter = registry.get(model_id)
        errors = adapter.validate(req.inputs)
        if errors:
            raise HTTPException(422, {"errors": errors})
        future = pool.submit(adapter.predict, req.inputs)
        try:
            return asdict(future.result(timeout=TIMEOUT_S))
        except FutureTimeout as e:
            raise HTTPException(504, "Prediction timed out") from e
        except ValueError as e:  # adapters raise ValueError for unusable input (e.g. not a chest X-ray)
            raise HTTPException(422, {"errors": [str(e)]}) from e

    return app


app = create_app()
