"""Upper-limb X-ray check: DenseNet-121 by Joyston Menezes (MIT), trained on Stanford MURA, served as the repo's ONNX.
Preprocessing matches the repo; the threshold comes from tools/evaluate_bone.py.
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

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "joystonmenezes_xray_vision"
MIN_SIDE = 200
MAX_COLOUR = 12.0  # mean channel difference: X-rays are grey, so a colour photo is refused


def prep(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    s = 256 / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
    canvas = Image.new("RGB", (256, 256))
    canvas.paste(img, ((256 - img.width) // 2, (256 - img.height) // 2))
    return np.asarray(canvas, dtype=np.float32)[16:240, 16:240][np.newaxis] / 255.0


class Adapter(ModelAdapter):
    id = "joystonmenezes_xray_vision"

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
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValueError("the image could not be read") from e
        if min(img.size) < MIN_SIDE:
            raise ValueError("the image is too small; please upload a clearer one")
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        if float(np.abs(rgb[..., 0] - rgb[..., 2]).mean()) > MAX_COLOUR:
            raise ValueError("this looks like a colour photo, not an X-ray. Please upload the X-ray image")
        p = float(self.session.run(["densenet121/probability:0"], {"inputs:0": prep(img)})[0][0, 1])
        t = self.meta["threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))  # scaled so 50% is the flagging point
        label = "Possible abnormality" if p >= t else "No clear abnormality"
        warnings = ["This check reads arm and hand X-rays only: shoulder, upper arm, elbow, forearm, wrist, hand and fingers."]
        return Prediction(model_id=self.id, label=label, warnings=warnings,
                          probabilities={"Possible abnormality": round(shown, 3), "No clear abnormality": round(1 - shown, 3)})
