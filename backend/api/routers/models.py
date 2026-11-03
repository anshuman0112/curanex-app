import json

from fastapi import APIRouter

from api.core.config import get_settings

router = APIRouter()


@router.get("/models")
def models():
    path = get_settings().registry_path
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"models": []}
