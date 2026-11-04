import json

from fastapi import APIRouter

from api.core.config import get_settings

router = APIRouter()


@router.get("/models")
def models():
    path = get_settings().registry_path
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
        for m in registry["models"]:
            m.pop("audit", None)  # build-time detail; keeps the response small for the app
        return registry
    return {"models": []}
