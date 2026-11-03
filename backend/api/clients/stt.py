"""Speech-to-text. Implemented in BACKEND_PLAN phase B5; until then /transcribe answers 503."""
from api.core.config import get_settings


class STTNotConfigured(Exception):
    pass


def transcribe(audio: bytes, filename: str) -> dict:
    s = get_settings()
    if s.stt_provider == "openai" and not s.openai_api_key:
        raise STTNotConfigured("Set OPENAI_API_KEY (or STT_PROVIDER=local) on the server.")
    raise STTNotConfigured("Speech-to-text is not implemented yet (BACKEND_PLAN phase B5).")
