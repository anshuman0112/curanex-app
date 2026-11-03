import pytest
from fastapi.testclient import TestClient

from api.core.config import get_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API client with no LLM/STT keys (env overrides backend/.env) and uploads in a temp dir."""
    for key in ("LLM_API_KEY", "AI_GATEWAY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))
    get_settings.cache_clear()
    from api.main import create_app

    yield TestClient(create_app())
    get_settings.cache_clear()
