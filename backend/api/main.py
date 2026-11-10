"""Curanex API app factory. Routes keep the paths the app already calls (/health /chat /upload /models /transcribe)."""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import get_settings
from api.core.logging import access_log, setup_logging
from api.orchestrator.retriever import warm_up
from api.routers import alerts, chat, health, models, uploads, voice


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=warm_up, daemon=True).start()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    app = FastAPI(title="Curanex API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])
    app.middleware("http")(access_log)
    for r in (health, chat, uploads, models, voice, alerts):
        app.include_router(r.router)
    return app


app = create_app()
