"""Nutrition: /v1/food-photo (photo -> dishes + nutrients), /v1/nutrition/targets and /v1/meal-plan (from the profile)."""
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api import cloud, nutrition
from api.auth.deps import optional_user

router = APIRouter(tags=["nutrition"])
MAX_BYTES = 12 * 1024 * 1024


class ProfileIn(BaseModel):
    profile: dict[str, Any] = {}


@router.post("/food-photo")
@router.post("/v1/food-photo")
async def food_photo(file: UploadFile = File(...), user: dict | None = Depends(optional_user)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=415, detail="Photo is larger than 12 MB.")
    try:
        out = await run_in_threadpool(nutrition.analyse_food, data)
    except OSError as e:
        raise HTTPException(status_code=415, detail="Please upload a JPEG, PNG or WebP photo.") from e
    if user and out.get("is_food"):
        import uuid
        name = f"meal-{uuid.uuid4().hex[:12]}.jpg"
        cloud.save_file(user["id"], name, data, "image/jpeg")
        cloud.save("meals", {"user_id": user["id"], "user_email": user["email"], "file_path": f"{user['id']}/{name}", **out})
    return out


@router.post("/v1/nutrition/targets")
def nutrition_targets(body: ProfileIn) -> dict:
    return nutrition.targets(body.profile)


@router.post("/meal-plan")
@router.post("/v1/meal-plan")
def meal_plan(body: ProfileIn) -> dict:
    return nutrition.meal_plan(body.profile)
