from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api import voice_check
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


MAX_AUDIO = 5 * 1024 * 1024


@router.get("/voice-check")
def voice_check_info():
    return {"available": voice_check.available(), "notice": voice_check.NOTICE}


@router.post("/voice-check")
async def voice_check_run(file: UploadFile = File(...), sex: str | None = Form(None)):
    """Experimental Parkinson's voice screening. The recording is measured and deleted; nothing is stored."""
    audio = await file.read()
    if len(audio) > MAX_AUDIO:
        raise HTTPException(status_code=413, detail="The recording is too long. About 5 seconds is enough.")
    try:
        return voice_check.check(audio, sex)
    except voice_check.VoiceCheckError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
