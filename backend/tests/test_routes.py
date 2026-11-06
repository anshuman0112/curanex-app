"""The app depends on these routes and on the /chat "blocks" contract (ARCHITECTURE.md §6)."""
from unittest.mock import patch

import main
from api.core.config import get_settings


def test_entrypoint_still_exports_app():
    assert main.app.title == "Curanex API"


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_chat_red_flag_escalates_without_llm(client):
    with patch("api.orchestrator.loop.call_llm") as llm:
        r = client.post("/chat", json={"session_id": "s1", "message": "crushing chest pain spreading to my left arm"})
    assert r.status_code == 200
    assert r.json() == {"blocks": [{"type": "escalate", "reason": "Possible heart attack"}]}
    llm.assert_not_called()


def test_chat_without_llm_key_returns_text_block(client):
    r = client.post("/chat", json={"session_id": "s1", "message": "what's a healthy breakfast"})
    blocks = r.json()["blocks"]
    assert len(blocks) == 1 and blocks[0]["type"] == "text"
    assert "LLM_API_KEY" in blocks[0]["text"]


def test_chat_passes_context_and_post_checks_reply(client):
    with patch("api.orchestrator.loop.call_llm", return_value="Take 500 mg of it.") as llm:
        r = client.post("/chat", json={"session_id": "s1", "message": "headache", "via": "voice",
                                       "context": {"profile": {"age": "34"}}})
    msgs, system = llm.call_args.args
    assert msgs[-1] == {"role": "user", "content": "headache"}
    assert '"age": "34"' in system and "spoken" in system
    text = r.json()["blocks"][0]["text"]
    assert "500 mg" not in text and "doctor" in text.lower()


def test_upload_returns_image_id(client):
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG")
    r = client.post("/upload", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")})
    image_id = r.json()["image_id"]
    assert (get_settings().uploads_dir / f"{image_id}.jpg").exists()


def test_models_returns_registry(client):
    models = client.get("/models").json()["models"]
    assert sum(m["wave"] != "replacement" for m in models) == 126  # the original list, plus models added to replace failures
    assert {m["wave"] for m in models} == {"wave-1", "wave-2", "backlog", "hospital-only", "replacement"}


def test_transcribe_not_configured_is_503(client):
    r = client.post("/transcribe", files={"file": ("a.webm", b"\x00", "audio/webm")})
    assert r.status_code == 503
