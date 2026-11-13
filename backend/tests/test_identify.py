"""Upload anything: the vision model only recognises the file; reports go to the report reader, images to the matching live model."""
import io
from unittest.mock import patch

from PIL import Image

from api import vision


def _png() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(b, format="PNG")
    return b.getvalue()


def test_routes_each_kind_to_its_model():
    assert vision.route("chest x-ray") == "mlmed_torchxrayvision"
    assert vision.route("hand or arm x-ray") == "joystonmenezes_xray_vision"
    assert vision.route("brain mri") == "brain_mri_check"
    assert vision.route("ultrasound", "breast") == "breast_ultrasound_check"
    assert vision.route("ultrasound", "abdomen") is None  # only a breast ultrasound model exists
    assert vision.route("ecg") is None and vision.route("ct scan") is None
    assert vision.is_document("lab report")


def test_image_goes_to_model(client):
    seen = {"kind": "chest x-ray", "body_part": "chest", "has_text": False, "findings": ["opacity in right lower zone"]}
    with patch("api.vision.identify", return_value=seen):
        body = client.post("/v1/identify", files={"file": ("x.png", _png(), "image/png")}).json()
    assert body["route"] == "image" and body["model_id"] == "mlmed_torchxrayvision" and body["image_id"]


def test_ecg_has_no_model_but_keeps_the_ai_reading(client):
    seen = {"kind": "ecg", "body_part": "heart", "has_text": True, "findings": ["regular rhythm"]}
    with patch("api.vision.identify", return_value=seen):
        body = client.post("/v1/identify", files={"file": ("e.png", _png(), "image/png")}).json()
    assert body["route"] == "image" and body["model_id"] is None and body["findings"] == ["regular rhythm"]


def test_lab_report_photo_goes_to_report_reader(client):
    seen = {"kind": "lab report", "body_part": "", "has_text": True, "findings": []}
    with patch("api.vision.identify", return_value=seen), patch("api.routers.uploads.read_report", return_value="Haemoglobin 10.2 g/dL\nHDL cholesterol 34 mg/dL"):
        body = client.post("/v1/identify", files={"file": ("r.png", _png(), "image/png")}).json()
    assert body["route"] == "report" and body["values"]["hdl"] == 34.0


def test_vision_down_is_not_an_error(client):
    with patch("api.vision.httpx.post", side_effect=OSError("offline")):
        body = client.post("/v1/identify", files={"file": ("x.png", _png(), "image/png")}).json()
    assert body["kind"] == "unknown" and body["model_id"] is None
