"""Image upload. Type/size checks and EXIF stripping arrive with /v1/uploads in phase B2."""
import uuid

from fastapi import APIRouter, File, UploadFile

from api.core.config import get_settings

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    uploads = get_settings().uploads_dir
    uploads.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex
    (uploads / f"{image_id}.jpg").write_bytes(await file.read())
    return {"image_id": image_id}
