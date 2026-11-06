"""Shared logic for multi-disease image models trained with training/train_folder_head.py (ONNX, nothing to unpickle)."""
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from workers.common.adapter import ModelAdapter, Prediction

MODELS = Path(__file__).resolve().parent / "models" / "multi"


class MultiHeadAdapter(ModelAdapter):
    """Subclasses set `name`, the plain `titles`, the `clear` class meaning nothing was found, and the image checks."""

    name = ""
    titles: dict = {}
    clear = ""
    found_label = "Possible {}"
    clear_label = "No clear signs found"
    min_side = 200
    grey = None  # True needs a grey image, False a colour one, None takes either
    min_colour = 6.0  # below this mean channel difference an image counts as grey
    wrong_image = "this doesn't look like the right kind of image"

    def load(self) -> None:
        self.meta = json.loads((MODELS / f"{self.name}.json").read_text())
        path = MODELS / f"{self.name}.onnx"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.meta["onnx_sha256"]:
            raise RuntimeError(f"{self.name}.onnx SHA-256 mismatch")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def validate(self, inputs: dict) -> list[str]:
        return [] if inputs.get("image_b64") else ["image_b64: missing"]

    def prep(self, img: Image.Image) -> np.ndarray:
        s = 256 / min(img.size)
        img = img.resize((max(224, round(img.width * s)), max(224, round(img.height * s))), Image.BILINEAR)
        left, top = (img.width - 224) // 2, (img.height - 224) // 2
        a = np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255
        mean, std = np.array(self.meta["mean"], dtype=np.float32), np.array(self.meta["std"], dtype=np.float32)
        return ((a - mean) / std).transpose(2, 0, 1)[np.newaxis]

    def check(self, rgb: np.ndarray) -> None:
        """Extra image check for a subclass. Raises ValueError to refuse the image."""

    def predict(self, inputs: dict) -> Prediction:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(inputs["image_b64"]))).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValueError("the image could not be read") from e
        if min(img.size) < self.min_side:
            raise ValueError("the image is too small; please upload a clearer one")
        rgb = np.asarray(img, dtype=np.float32)
        colour = float(np.abs(rgb[..., 0] - rgb[..., 2]).mean())
        if (self.grey is True and colour > 12) or (self.grey is False and colour < self.min_colour):
            raise ValueError(self.wrong_image)
        self.check(rgb)
        p = self.session.run(None, {"image": self.prep(img)})[0][0]
        labels = self.meta["labels"]
        scores = {self.titles.get(k, k): round(float(v), 3) for k, v in zip(labels, p, strict=True)}
        clear_i = labels.index(self.clear) if self.clear else -1
        if clear_i >= 0 and p[clear_i] >= self.meta["thresholds"][clear_i]:
            return Prediction(model_id=self.id, label=self.clear_label, probabilities=scores)
        top = max((i for i in range(len(labels)) if i != clear_i), key=lambda i: p[i])
        label = self.found_label.format(self.titles.get(labels[top], labels[top]).lower())
        return Prediction(model_id=self.id, label=label, probabilities=scores)
