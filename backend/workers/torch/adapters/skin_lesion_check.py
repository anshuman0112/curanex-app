"""Skin lesion check: EfficientNet-B0 features with a logistic layer, trained by the Curanex team on HAM10000 (training/train_skin.py).
Flags lesions that look suspicious (melanoma, basal cell carcinoma, actinic keratosis). Trained on dermatoscope images.
"""
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from workers.common.adapter import ModelAdapter, Prediction

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "skin_lesion_check"
MIN_SIDE = 200
MIN_COLOUR = 6.0  # skin photos are in colour; X-rays and scans are grey


def prep(img: Image.Image, mean: list, std: list) -> np.ndarray:
    s = 256 / min(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    a = np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255
    return ((a - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)).transpose(2, 0, 1)[np.newaxis]


class Adapter(ModelAdapter):
    id = "skin_lesion_check"

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        path = MODEL_DIR / "model.onnx"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.meta["onnx_sha256"]:
            raise RuntimeError("model.onnx SHA-256 mismatch")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def validate(self, inputs: dict) -> list[str]:
        return [] if inputs.get("image_b64") else ["image_b64: missing"]

    def predict(self, inputs: dict) -> Prediction:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(inputs["image_b64"]))).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValueError("the image could not be read") from e
        if min(img.size) < MIN_SIDE:
            raise ValueError("the image is too small; please upload a clearer one")
        rgb = np.asarray(img, dtype=np.float32)
        if float(np.abs(rgb[..., 0] - rgb[..., 2]).mean()) < MIN_COLOUR:
            raise ValueError("this looks like a scan or X-ray. Please upload a colour photo of the skin")
        m = self.meta
        p = float(self.session.run(None, {"image": prep(img, m["mean"], m["std"])})[0].ravel()[0])
        t = m["threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))  # scaled so 50% is the flagging point
        label = "Looks suspicious, please see a skin doctor" if p >= t else "No clear warning signs"
        warnings = ["Trained on clinic dermatoscope images, so a phone photo may be less reliable. Show any changing mole to a doctor."]
        scores = {"Suspicious signs": round(shown, 3), "No clear signs": round(1 - shown, 3)}
        return Prediction(model_id=self.id, label=label, warnings=warnings, probabilities=scores)
