from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.reports import ReportRejected, find_values, read_report
from api.scans import ScanRejected, is_scan, scan_to_image
from api import cloud, vision
from api.auth.deps import optional_user
from api.uploads_store import UploadRejected, save_image

router = APIRouter()
SCAN_MAX_BYTES = 60 * 1024 * 1024  # a zipped series; the image kept from it is small


@router.post("/upload")
@router.post("/v1/uploads")
async def upload(file: UploadFile = File(...), user: dict | None = Depends(optional_user)):
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
    _keep_image(user, image_id, data, {})
    return {"image_id": image_id, "upload_id": image_id, **info}


def _keep_image(user: dict | None, image_id: str, data: bytes, seen: dict, model_id: str | None = None) -> None:
    if user:
        path = cloud.save_file(user["id"], f"{image_id}.jpg", data, "image/jpeg")
        cloud.save("uploads", {"user_id": user["id"], "user_email": user["email"], "image_id": image_id, "file_path": path, "kind": seen.get("kind"),
                               "body_part": seen.get("body_part"), "model_id": model_id, "findings": seen.get("findings")})


def _keep_report(user: dict | None, data: bytes, out: dict) -> None:
    if user:
        pdf = data[:5] == b"%PDF-"
        import uuid
        path = cloud.save_file(user["id"], f"report-{uuid.uuid4().hex[:12]}.{'pdf' if pdf else 'jpg'}", data, "application/pdf" if pdf else "image/jpeg")
        cloud.save("reports", {"user_id": user["id"], "user_email": user["email"], "file_path": path, "text": out.get("text"), "lab_values": out.get("values")})


@router.post("/report")
@router.post("/v1/reports")
async def report(file: UploadFile = File(...), user: dict | None = Depends(optional_user)):
    """Text of a PDF or photographed report. Kept in memory only."""
    data = await file.read()
    try:
        text = await run_in_threadpool(read_report, data)
    except ReportRejected as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    out = {"text": text, "values": find_values(text)}
    _keep_report(user, data, out)
    return out


@router.post("/identify")
@router.post("/v1/identify")
async def identify_upload(file: UploadFile = File(...), user: dict | None = Depends(optional_user)):
    """Upload anything. Reports/PDFs -> {"route": "report", text, values}. Images -> {"route": "image", image_id, kind, body_part,
    findings, model_id (live model that checks it, or null)}. A DICOM/zip scan is turned into one image first."""
    data = await file.read()
    if data[:5] == b"%PDF-":
        out = await _as_report(data)
        _keep_report(user, data, out)
        return out
    info: dict = {}
    try:
        if is_scan(data):
            if len(data) > SCAN_MAX_BYTES:
                raise UploadRejected("The scan is larger than 60 MB. Upload one series.")
            data, info = await run_in_threadpool(scan_to_image, data)
        seen = await run_in_threadpool(vision.identify, data)
        if vision.is_document(seen["kind"]) or (seen["kind"] in ("unknown", "other") and seen["has_text"]):
            out = await _as_report(data) | {"kind": seen["kind"]}
            _keep_report(user, data, out)
            return out
        image_id = save_image(data)
    except (UploadRejected, ScanRejected) as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    model_id = vision.route(seen["kind"], seen["body_part"])
    _keep_image(user, image_id, data, seen, model_id)
    return {"route": "image", "image_id": image_id, **seen, "model_id": model_id, **info}


async def _as_report(data: bytes) -> dict:
    try:
        text = await run_in_threadpool(read_report, data)
    except ReportRejected as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    return {"route": "report", "text": text, "values": find_values(text)}
