from fastapi import APIRouter, File, HTTPException, UploadFile

from api.uploads_store import UploadRejected, save_image

router = APIRouter()


@router.post("/upload")
@router.post("/v1/uploads")
async def upload(file: UploadFile = File(...)):
    try:
        image_id = save_image(await file.read())
    except UploadRejected as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    return {"image_id": image_id, "upload_id": image_id}
