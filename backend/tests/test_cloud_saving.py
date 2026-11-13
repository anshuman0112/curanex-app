"""Logged-in patients: chats, model results and uploads are saved to their own cloud rows; anonymous use saves nothing."""
import io
from unittest.mock import patch

from PIL import Image

from tests.test_auth_records import signup


def _png() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(b, format="PNG")
    return b.getvalue()


def test_chat_is_saved_only_for_logged_in_users(client):
    blocks = [{"type": "text", "text": "Drink fluids."}, {"type": "model_result", "model_id": "m", "model_name": "M", "label": "Lower risk", "flag": "green", "probabilities": {}}]
    with patch("api.routers.chat.orchestrate", return_value=blocks), patch("api.cloud.save") as save:
        client.post("/chat", json={"session_id": "s", "message": "hello"})
        assert save.call_count == 0
        headers, user = signup(client, "p@cloud.test", "patient")
        client.post("/chat", json={"session_id": "s", "message": "hello"}, headers=headers)
    tables = [c.args[0] for c in save.call_args_list]
    assert tables == ["chat_messages", "chat_messages", "model_results"]
    assert all(c.args[1]["user_id"] == user["id"] for c in save.call_args_list)


def test_upload_is_saved_to_the_users_folder(client):
    headers, user = signup(client, "u@cloud.test", "patient")
    with patch("api.cloud.save") as save, patch("api.cloud.save_file", return_value=f"{user['id']}/x.jpg") as save_file:
        r = client.post("/upload", files={"file": ("x.png", _png(), "image/png")}, headers=headers)
    assert r.status_code == 200
    assert save_file.call_args.args[0] == user["id"] and save.call_args.args[0] == "uploads"


def test_my_data_needs_login(client):
    assert client.get("/me/data").status_code == 401
