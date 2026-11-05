import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import experimental
from api.core.config import get_settings
from api.uploads_store import load_image

router = APIRouter()


class ExperimentalRun(BaseModel):
    image_id: str


@router.get("/models")
def models():
    path = get_settings().registry_path
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
        for m in registry["models"]:
            m.pop("audit", None)  # build-time detail; keeps the response small for the app
        return registry
    return {"models": []}


@router.get("/experimental")
def experimental_models():
    return {"models": experimental.catalog()}


@router.post("/experimental/{model_id}")
def run_experimental(model_id: str, body: ExperimentalRun):
    image = load_image(body.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    result = experimental.predict(model_id, image)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown experimental model")
    return result
