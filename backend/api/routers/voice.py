from fastapi import APIRouter, File, HTTPException, UploadFile

from api.clients.stt import STTFailed, STTNotConfigured, transcribe

router = APIRouter()


@router.post("/transcribe")
@router.post("/v1/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        return transcribe(await file.read(), file.filename or "audio.webm")
    except STTNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except STTFailed as e:
        raise HTTPException(status_code=502, detail="Speech-to-text provider error") from e
