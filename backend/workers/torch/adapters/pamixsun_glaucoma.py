"""Glaucoma check from a retinal (fundus) photo: Swin Transformer V2 by pamixsun (Apache-2.0), trained on REFUGE, served as ONNX.
Thresholds and scores come from tools/evaluate_glaucoma_swin.py. The image check matches the cataract adapter.
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

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "pamixsun_glaucoma"
SIZE = 512
MIN_SIDE = 200
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Adapter(ModelAdapter):
    id = "pamixsun_glaucoma"

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / "convert.json").read_text())
        self.threshold = json.loads((MODEL_DIR / "evaluation.json").read_text())["threshold"]
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
        r, g, b = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
        dark = float((rgb.mean(axis=2) < 15).mean())
        if not (r > g >= b - 5) or dark < 0.05:
            raise ValueError("this doesn't look like a retinal (fundus) photo from an eye clinic camera")
        a = np.asarray(img.resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float32) / 255
        logits = self.session.run(None, {"image": ((a - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]})[0][0]
        e = np.exp(logits - logits.max())
        p = float(e[1] / e.sum())
        t = self.threshold
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))  # scaled so 50% is the flagging point
        label = "Possible glaucoma signs" if p >= t else "No clear glaucoma signs"
        scores = {"Glaucoma signs": round(shown, 3), "No clear signs": round(1 - shown, 3)}
        return Prediction(model_id=self.id, label=label, probabilities=scores)
