"""Teeth photo check: YOLOv8m caries detector by AndreyGermanov (GPL-3.0, server-side only), served as ONNX.
The score for a photo is the model's highest caries or cavity confidence anywhere in it.
"""
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, UnidentifiedImageError

from workers.common.adapter import ModelAdapter, Prediction

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "andreygermanov_yolov8_caries_detector"
SIZE = 640
CLASSES = ["Caries", "Cavity", "Crack", "Tooth"]
MIN_SIDE = 200
MIN_COLOUR = 6.0  # mean channel difference: teeth photos are in colour; X-rays and screenshots of them are grey


def letterbox(img: Image.Image) -> tuple[np.ndarray, float, int, int]:
    """Ultralytics-style resize with grey padding. Returns [1,3,640,640] in 0-1, scale, pad x, pad y."""
    img = img.convert("RGB")
    scale = SIZE / max(img.size)
    w, h = round(img.width * scale), round(img.height * scale)
    pad_x, pad_y = (SIZE - w) // 2, (SIZE - h) // 2
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(img.resize((w, h), Image.BILINEAR), (pad_x, pad_y))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    return x, scale, pad_x, pad_y


def lesion_score(pred: np.ndarray) -> tuple[float, np.ndarray]:
    """pred: [1,8,N] (cx,cy,w,h + 4 class probabilities). Highest caries/cavity confidence, and per-class maxima."""
    probs = pred[0, 4:, :]
    return float(probs[:2].max()), probs.max(axis=1)


class Adapter(ModelAdapter):
    id = "andreygermanov_yolov8_caries_detector"

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
            img = Image.open(io.BytesIO(base64.b64decode(inputs["image_b64"])))
            img.load()
        except (ValueError, UnidentifiedImageError, OSError) as e:
            raise ValueError("the image could not be read") from e
        if min(img.size) < MIN_SIDE:
            raise ValueError(f"the image is too small ({img.width}x{img.height}); please upload a clearer one")
        rgb = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)
        if np.mean(np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])) < MIN_COLOUR:
            raise ValueError("this looks like a grey image or an X-ray, not a colour photo of teeth")
        x, _, _, _ = letterbox(img)
        pred = self.session.run(None, {"images": x})[0]
        score, per_class = lesion_score(pred)
        threshold = self.meta["threshold"]
        tooth_seen = float(per_class[3]) >= 0.25
        warnings = [] if tooth_seen else ["The model could not find teeth clearly in this photo, so this result is unreliable."]
        found = score >= threshold
        scaled = score / (threshold * 2) if score < threshold else 1 - (1 - score) / ((1 - threshold) * 2)
        best = int(np.argmax(pred[0, 4:6, :].max(axis=1)[:2]))
        i = int(pred[0, 4 + best].argmax())
        cx, cy, w, h = pred[0, :4, i]
        canvas = Image.fromarray((x[0].transpose(1, 2, 0) * 255).astype(np.uint8))
        if found:
            ImageDraw.Draw(canvas).rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=(255, 140, 0), width=4)
        buf = io.BytesIO()
        canvas.resize((448, 448)).save(buf, format="PNG")
        label = "Possible tooth decay spotted" if found else "No clear decay spotted"
        return Prediction(model_id=self.id, label=label, warnings=warnings,
                          probabilities={"Tooth decay signs": round(float(scaled), 3)},
                          explanation={"heatmap_finding": "Tooth decay signs" if found else "No marked area",
                                       "heatmap_png_b64": base64.b64encode(buf.getvalue()).decode()})
