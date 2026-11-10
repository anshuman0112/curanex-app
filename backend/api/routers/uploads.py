from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.reports import ReportRejected, find_values, read_report
from api.scans import ScanRejected, is_scan, scan_to_image
from api.uploads_store import UploadRejected, save_image

router = APIRouter()
SCAN_MAX_BYTES = 60 * 1024 * 1024  # a zipped series; the image kept from it is small


@router.post("/upload")
@router.post("/v1/uploads")
async def upload(file: UploadFile = File(...)):
    """A photo, or a DICOM file / zipped series. A scan is turned into one image and its header (name, ID, hospital) dropped."""
    data = await file.read()
    info: dict = {}
    try:
        if is_scan(data):
            if len(data) > SCAN_MAX_BYTES:
                raise UploadRejected("The scan is larger than 60 MB. Upload one series.")
            data, info = await run_in_threadpool(scan_to_image, data)
        image_id = save_image(data)
    except (UploadRejected, ScanRejected) as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    return {"image_id": image_id, "upload_id": image_id, **info}


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
