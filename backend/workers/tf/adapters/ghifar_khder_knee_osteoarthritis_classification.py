"""Knee X-ray arthritis check: Ghifar Khder's Kellgren-Lawrence grading CNN, converted to ONNX (no Keras at runtime).
The app reports moderate-or-worse (KL 2+) vs not: exact grade accuracy is only 69% on the held-out images.
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

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "ghifar_khder_knee_osteoarthritis_classification"
SIZE = 224
MIN_SIDE = 150
MAX_COLOUR = 18.0  # mean channel difference: real X-rays (and photos of them) are close to grey
GRADES = ["None", "Doubtful", "Mild", "Moderate", "Severe"]


class Adapter(ModelAdapter):
    id = "ghifar_khder_knee_osteoarthritis_classification"

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
        except (ValueError, UnidentifiedImageError, OSError) as e:
            raise ValueError("the image could not be read") from e
        if min(img.size) < MIN_SIDE:
            raise ValueError(f"the image is too small ({img.width}x{img.height}); please upload a clearer one")
        small = np.asarray(img.resize((128, 128)), dtype=np.float32)
        if np.mean(np.abs(small[..., 0] - small[..., 1]) + np.abs(small[..., 1] - small[..., 2])) > MAX_COLOUR:
            raise ValueError("this looks like a colour photo, not a knee X-ray")
        x = np.asarray(img.resize((SIZE, SIZE)), dtype=np.float32)[np.newaxis] / 255.0
        p = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0]
        arthritis = float(p[2:].sum())
        label = "Possible moderate arthritis signs" if arthritis >= 0.5 else "No clear moderate arthritis signs"
        return Prediction(model_id=self.id, label=label, probabilities={"Moderate or worse": round(arthritis, 3)},
                          explanation={"grade_estimate": GRADES[int(p.argmax())]},
                          warnings=["This checks one knee on a front-view X-ray. Other views may give wrong results."])
