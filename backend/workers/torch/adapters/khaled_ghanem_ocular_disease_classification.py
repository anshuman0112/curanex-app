"""Eye check from a retinal (fundus) photo: Khaled Ghanem's cataract and glaucoma models, converted to ONNX.
Uses the preprocessing from the repo's notebooks; polarity, scale and thresholds are set in tools/evaluate_eye.py.
"""
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from workers.common.adapter import ModelAdapter, Prediction

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "khaled_ghanem_ocular_disease_classification"
TITLES = {"cataract": "Cataract signs", "glaucoma": "Glaucoma signs"}
SIZE = 224
MIN_SIDE = 200


def crop_black(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Remove the black border around a fundus photo."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    return img[np.ix_(mask.any(1), mask.any(0))]


def prep_glaucoma(img: np.ndarray) -> np.ndarray:
    """Repo notebook: crop, CLAHE on the L channel, resize, 0-1."""
    img = crop_black(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return cv2.resize(img, (SIZE, SIZE)).astype(np.float32)[np.newaxis] / 255.0


def prep_cataract(img: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(img, (SIZE, SIZE)).astype(np.float32)[np.newaxis] / scale


class Adapter(ModelAdapter):
    id = "khaled_ghanem_ocular_disease_classification"

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        self.sessions = {}
        passed = [n for n in TITLES if "threshold" in self.meta.get(n, {}) and "rejected" not in self.meta[n]]
        for name in passed:
            path = MODEL_DIR / f"{name}.onnx"
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.meta[name]["onnx_sha256"]:
                raise RuntimeError(f"{name}.onnx SHA-256 mismatch")
            self.sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def validate(self, inputs: dict) -> list[str]:
        return [] if inputs.get("image_b64") else ["image_b64: missing"]

    def raw(self, name: str, img: np.ndarray) -> float:
        m = self.meta[name]
        x = prep_cataract(img, m["scale"]) if name == "cataract" else prep_glaucoma(img)
        s = self.sessions[name]
        p = float(s.run(None, {s.get_inputs()[0].name: x})[0].ravel()[0])
        return 1 - p if m["inverted"] else p

    def predict(self, inputs: dict) -> Prediction:
        data = np.frombuffer(base64.b64decode(inputs["image_b64"]), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR, as the notebooks read it
        if img is None:
            raise ValueError("the image could not be read")
        if min(img.shape[:2]) < MIN_SIDE:
            raise ValueError("the image is too small; please upload a clearer one")
        b, g, r = [img[..., i].astype(np.float32).mean() for i in range(3)]
        dark = float((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 15).mean())
        if not (r > g >= b - 5) or dark < 0.05:
            raise ValueError("this doesn't look like a retinal (fundus) photo from an eye clinic camera")
        scores = {}
        for name in self.sessions:
            p = self.raw(name, img)
            t = self.meta[name]["threshold"]
            scores[TITLES[name]] = round(p / (t * 2) if p < t else 1 - (1 - p) / ((1 - t) * 2), 3)
        flagged = [k for k, v in scores.items() if v >= 0.5]
        label = f"Possible {flagged[0].lower()}" if flagged else "No clear signs found"
        return Prediction(model_id=self.id, label=label, probabilities=scores, explanation={"flagged": flagged})
