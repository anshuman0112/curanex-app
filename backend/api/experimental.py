"""Experimental models: small ONNX networks with measured scores, for demonstrations only.

They are never offered to model search or to patients. Results go only to the Experimental section of the Model Library,
under a notice. Each is trained on 28x28 images, so a result means little for a real patient's photo.
"""
import io
import json
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS = Path(__file__).resolve().parents[1] / "workers" / "torch" / "models" / "medmnist"
SIZE = 28
NOTICE = "Research demo on tiny test images. Not a medical result and not for patient use."
_sessions: dict[str, ort.InferenceSession] = {}
_lock = threading.Lock()


def catalog() -> list[dict]:
    out = []
    for path in sorted(MODELS.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if (MODELS / f"{path.stem}.onnx").exists():
            out.append({k: meta[k] for k in ("id", "title", "labels", "test")} | {"notice": NOTICE})
    return out


def predict(model_id: str, image: bytes) -> dict | None:
    meta_path = MODELS / f"{model_id}.json"
    if not model_id.isalnum() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with _lock:
        if model_id not in _sessions:
            _sessions[model_id] = ort.InferenceSession(str(MODELS / f"{model_id}.onnx"), providers=["CPUExecutionProvider"])
    img = Image.open(io.BytesIO(image)).convert("L" if meta["channels"] == 1 else "RGB").resize((SIZE, SIZE), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255
    a = a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
    logits = _sessions[model_id].run(None, {"image": ((a - 0.5) / 0.5)[np.newaxis]})[0][0]
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    order = np.argsort(-p)[:3]
    top = [[meta["labels"][int(i)], round(float(p[i]), 3)] for i in order]
    return {"label": top[0][0], "top": top, "notice": NOTICE}
