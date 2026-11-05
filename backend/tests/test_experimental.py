"""Experimental models stay out of patient paths and give a labelled result."""
import io

from PIL import Image

from api import experimental
from api.registry import get_routable


def test_catalog_and_run_carry_the_notice(client):
    models = client.get("/experimental").json()["models"]
    assert models and all("not for patient" in m["notice"].lower() for m in models)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 80)).save(buf, "PNG")
    image_id = client.post("/upload", files={"file": ("a.png", buf.getvalue(), "image/png")}).json()["image_id"]
    r = client.post(f"/experimental/{models[0]['id']}", json={"image_id": image_id}).json()
    assert r["label"] in models[0]["labels"] and "not for patient" in r["notice"].lower()


def test_unknown_model_or_path_tricks_are_refused(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "PNG")
    image_id = client.post("/upload", files={"file": ("a.png", buf.getvalue(), "image/png")}).json()["image_id"]
    assert client.post("/experimental/nothing", json={"image_id": image_id}).status_code == 404
    assert experimental.predict("../secret", b"") is None


def test_experimental_models_are_not_in_patient_search(client):
    ids = {m["id"] for m in client.get("/experimental").json()["models"]}
    live = {m["id"] for m in client.get("/models").json()["models"] if m.get("integrated")}
    assert not ids & live
    assert all(get_routable(i) is None for i in ids)
