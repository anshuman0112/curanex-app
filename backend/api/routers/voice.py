from fastapi import APIRouter, File, HTTPException, UploadFile

from api.clients.stt import STTNotConfigured, transcribe

router = APIRouter()


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        return transcribe(await file.read(), file.filename or "audio")
    except STTNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
