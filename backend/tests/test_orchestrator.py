"""Model platform (phase B2): model search, tool loop, block contract, safety on every path."""
import importlib.util
import io
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.core.config import get_settings
from api.orchestrator import retriever
from api.safety import post_check
from workers.common.server import create_app as create_worker

PROFILE = {"age": "34", "sex": "Male", "bmi": 27.1}
needs_embedder = pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None,
                                    reason="sentence-transformers not installed")

# The app's Block type (src/lib/api.ts): required keys per block type
CONTRACT = {
    "text": {"text"},
    "escalate": {"reason"},
    "form_request": {"model_id", "title", "fields"},
    "upload_request": {"model_id", "kind", "instructions"},
    "model_result": {"model_id", "model_name", "label", "probabilities", "trained_on", "limitations"},
}


def assert_contract(blocks):
    for b in blocks:
        assert CONTRACT[b["type"]] <= set(b), b
        if b["type"] == "form_request":
            for f in b["fields"]:
                assert {"key", "label", "type"} <= set(f) and f["type"] in ("number", "text")
        if b["type"] == "model_result":
            assert b["disclaimer"] == "Not a diagnosis" and b["credit"]


@pytest.fixture
def api(client, monkeypatch, tmp_path):
    """API with the dummy model searchable and worker calls served in-process."""
    monkeypatch.setenv("ENABLE_DUMMY_MODEL", "true")
    monkeypatch.setenv("EMBEDDINGS_CACHE", str(tmp_path / "emb.npz"))
    get_settings.cache_clear()
    retriever.set_embedder(None)
    worker = TestClient(create_worker("tabular"))

    def fake_predict(worker_type, model_id, inputs):
        from api.clients.workers import WorkerError
        r = worker.post(f"/predict/{model_id}", json={"inputs": inputs})
        if r.status_code != 200:
            raise WorkerError("; ".join(r.json()["detail"]["errors"]), r.status_code)
        return r.json()

    with patch("api.orchestrator.loop.workers.predict", side_effect=fake_predict):
        yield client
    retriever.set_embedder(None)


def chat(c, **body):
    r = c.post("/chat", json={"session_id": "s1", "context": {"profile": PROFILE}, **body})
    assert r.status_code == 200
    blocks = r.json()["blocks"]
    assert_contract(blocks)
    return blocks


@needs_embedder
def test_search_routes_dummy_and_not_breakfast(api):
    assert [m["id"] for m, _ in retriever.search("please run the echo test model")] == ["dummy_echo"]
    assert retriever.search("what's a healthy breakfast?") == []


@needs_embedder
def test_no_llm_matching_phrase_asks_for_inputs_with_profile_prefill(api):
    blocks = chat(api, message="run the echo test model")
    assert [b["type"] for b in blocks] == ["text", "form_request"]
    form = blocks[1]
    assert form["model_id"] == "dummy_echo" and form["prefilled"] == {"age": "34"}
    assert [f["key"] for f in form["fields"]] == ["echo_score"]


@needs_embedder
def test_no_model_for_general_question(api):
    blocks = chat(api, message="what's a healthy breakfast?")
    assert [b["type"] for b in blocks] == ["text"]


def test_form_follow_up_runs_model_and_explains(api):
    blocks = chat(api, message="", model_id="dummy_echo", form_values={"echo_score": "8"})
    assert [b["type"] for b in blocks] == ["model_result", "text"]
    assert blocks[0]["label"] == "Higher" and blocks[0]["probabilities"]["Higher"] == 0.8
    assert "not a diagnosis" in blocks[1]["text"].lower()


def test_form_follow_up_with_bad_value_reports_error(api):
    blocks = chat(api, message="", model_id="dummy_echo", form_values={"echo_score": "50"})
    assert blocks[0]["type"] == "text" and "out of range" in blocks[0]["text"]


def test_unknown_model_follow_up_is_refused(api):
    blocks = chat(api, message="", model_id="not_a_model", form_values={"x": "1"})
    assert blocks[0]["type"] == "text" and "isn't available" in blocks[0]["text"]


def _tool_call(name, args):
    return {"content": "", "tool_calls": [{"id": "c1", "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_tool_loop_runs_model_then_explains(api):
    dummy = retriever.routable_models()[-1]
    replies = [_tool_call("run_dummy_echo", {"echo_score": 7}), {"content": "Your echo estimate is higher."}]
    with patch("api.orchestrator.loop.search", return_value=[(dummy, 0.6)]), \
         patch("api.orchestrator.loop.llm_configured", return_value=True), \
         patch("api.orchestrator.loop.chat_completion", side_effect=replies) as llm:
        blocks = chat(api, message="run the echo test", via="voice")
    assert [b["type"] for b in blocks] == ["model_result", "text"]
    assert blocks[0]["probabilities"]["Higher"] == 0.7
    tools = [t["function"]["name"] for t in llm.call_args_list[0].args[1]]
    assert tools == ["run_dummy_echo", "request_user_input", "request_image"]
    second = llm.call_args_list[1].args[0]
    assert second[-1]["role"] == "tool" and "Higher" in second[-1]["content"]
    assert "spoken" in second[0]["content"]


def test_tool_loop_request_user_input_becomes_form(api):
    dummy = retriever.routable_models()[-1]
    reply = _tool_call("request_user_input", {"model_id": "dummy_echo", "fields": ["echo_score"], "reason": "Quick check"})
    with patch("api.orchestrator.loop.search", return_value=[(dummy, 0.6)]), \
         patch("api.orchestrator.loop.llm_configured", return_value=True), \
         patch("api.orchestrator.loop.chat_completion", return_value=reply):
        blocks = chat(api, message="echo test please")
    assert [b["type"] for b in blocks] == ["text", "form_request"]
    assert blocks[1]["prefilled"] == {"age": "34"}


def test_tool_loop_plain_answer_when_llm_declines(api):
    dummy = retriever.routable_models()[-1]
    with patch("api.orchestrator.loop.search", return_value=[(dummy, 0.4)]), \
         patch("api.orchestrator.loop.llm_configured", return_value=True), \
         patch("api.orchestrator.loop.chat_completion", return_value={"content": "You definitely have diabetes."}):
        blocks = chat(api, message="echo?")
    assert blocks == [{"type": "text", "text": post_check("You definitely have diabetes.")}]
    assert "definitely have diabetes" not in blocks[0]["text"]


def test_red_flag_never_reaches_search_or_llm(api):
    with patch("api.orchestrator.loop.search") as s, patch("api.orchestrator.loop.chat_completion") as llm, \
         patch("api.orchestrator.loop.call_llm") as llm2:
        blocks = chat(api, message="I can't breathe and my lips are turning blue")
    assert blocks == [{"type": "escalate", "reason": "Severe breathing difficulty"}]
    s.assert_not_called(), llm.assert_not_called(), llm2.assert_not_called()


def test_post_check_softens_diagnosis_wording_only():
    assert "you may have signs of" in post_check("Based on this you have type 2 diabetes.")
    assert "you have been very thirsty" in post_check("I see you have been very thirsty lately; see a doctor.")


def _img(fmt="PNG", exif=False):
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), "gray")
    if exif:
        e = Image.Exif()
        e[0x010F] = "PhoneMaker"  # Make
        img.save(buf, format="JPEG", exif=e)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def test_upload_validates_and_strips_exif(api):
    r = api.post("/v1/uploads", files={"file": ("a.jpg", _img(exif=True), "image/jpeg")})
    assert r.status_code == 200
    saved = Image.open(get_settings().uploads_dir / f"{r.json()['image_id']}.jpg")
    assert not saved.getexif()
    assert api.post("/upload", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415


def test_upload_size_limit(api, monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "100")
    get_settings.cache_clear()
    assert api.post("/upload", files={"file": ("a.png", _img(), "image/png")}).status_code == 415
