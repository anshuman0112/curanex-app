from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.reports import ReportRejected, find_values, read_report
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


@router.post("/report")
@router.post("/v1/reports")
async def report(file: UploadFile = File(...)):
    """Text of a PDF or photographed report. Kept in memory only."""
    data = await file.read()
    try:
        text = await run_in_threadpool(read_report, data)
    except ReportRejected as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    return {"text": text, "values": find_values(text)}
