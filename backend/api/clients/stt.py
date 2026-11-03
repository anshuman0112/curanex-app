"""Speech-to-text via an OpenAI-compatible /audio/transcriptions endpoint (Groq Whisper is free)."""
import httpx

from api.core.config import get_settings


class STTNotConfigured(Exception):
    pass


class STTFailed(Exception):
    pass


def transcribe(audio: bytes, filename: str) -> dict:
    s = get_settings()
    base, key = s.stt_endpoint
    if not key:
        raise STTNotConfigured("Set LLM_API_KEY (Groq) or STT_API_KEY on the server to enable voice.")
    try:
        r = httpx.post(f"{base.rstrip('/')}/audio/transcriptions", headers={"Authorization": f"Bearer {key}"},
                       files={"file": (filename, audio)}, data={"model": s.stt_model, "response_format": "verbose_json"},
                       timeout=60)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise STTFailed(type(e).__name__) from e
    j = r.json()
    return {"text": j.get("text", "").strip(), "language": j.get("language")}
