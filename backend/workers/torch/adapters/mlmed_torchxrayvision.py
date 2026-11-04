"""Chest X-ray findings: torchxrayvision DenseNet-121 "all" weights, served as ONNX (no pickle at runtime).

Scores are the library's operating-point normalised values: 0.5 is the point chosen per finding on the validation
data, so a score above 0.5 means "flagged". Several findings can be flagged at once (multi-label).
Heatmaps are class activation maps of the top finding, exported in the ONNX graph.
Credit: Cohen et al., https://github.com/mlmed/torchxrayvision (Apache-2.0).
"""
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from skimage.transform import resize

from workers.common.adapter import ModelAdapter, Prediction

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "mlmed_torchxrayvision"
FLAG_AT = 0.5
TOP_N = 5
MIN_SIDE = 200
MAX_COLOUR = 18.0  # mean channel difference: real X-rays (and photos of them) are close to grey
READABLE = {"Lung Opacity": "Lung opacity", "Enlarged Cardiomediastinum": "Enlarged heart shadow", "Effusion": "Fluid around the lung",
            "Pleural_Thickening": "Pleural thickening", "Cardiomegaly": "Enlarged heart", "Lung Lesion": "Lung lesion",
            "Infiltration": "Infiltration", "Consolidation": "Consolidation", "Pneumothorax": "Collapsed lung (pneumothorax)"}


# Layers trained by the Curanex team on this model's features (tools/train_xrv_*). Each has its own measured metrics.
ADULT_HEADS = {"Abnormal X-ray": "abnormal_adult_head.json", "Lung opacity signs": "lung_adult_head.json"}
CHILD_HEADS = {"Pneumonia signs": "pneumonia_child_head.json"}
CHILD_AGE = 13


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def op_norm(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Same as torchxrayvision.models.op_norm: maps each finding's operating point to 0.5."""
    out = np.full_like(probs, 0.5)
    for i, t in enumerate(thresholds):
        if np.isnan(t):
            continue
        p = probs[:, i]
        out[:, i] = np.where(p < t, p / (t * 2), 1 - (1 - p) / ((1 - t) * 2))
    return out


def preprocess(img: Image.Image) -> np.ndarray:
    """Library pipeline: grey, scale to [-1024, 1024], centre crop to a square, resize to 224. Returns [1,1,224,224]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).mean(axis=2)
    arr = (2 * (arr / 255.0) - 1.0) * 1024
    h, w = arr.shape
    side = min(h, w)
    top, left = h // 2 - side // 2, w // 2 - side // 2
    arr = arr[top:top + side, left:left + side]
    arr = resize(arr, (224, 224), mode="constant", preserve_range=True).astype(np.float32)
    return arr[np.newaxis, np.newaxis, ...]


def check_image(img: Image.Image) -> tuple[list[str], list[str]]:
    """(reasons to refuse, warnings). Simple checks only: they can't prove an image is a chest X-ray."""
    w, h = img.size
    rgb = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)
    colour = float(np.mean(np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])))
    errors, warnings = [], []
    if min(w, h) < MIN_SIDE:
        errors.append(f"the image is too small ({w}x{h}); please upload a clearer one")
    if colour > MAX_COLOUR:
        errors.append("this looks like a colour photo, not a chest X-ray")
    if not 0.6 <= w / h <= 1.6:
        errors.append("the shape doesn't look like a chest X-ray; please crop it to the X-ray")
    if rgb.std() < 25:
        warnings.append("The image has low contrast, so this result is less reliable.")
    if colour > MAX_COLOUR / 3 and not errors:
        warnings.append("The image has some colour, which suggests a photo of a screen or film; results are less reliable.")
    return errors, warnings


def heatmap_png(x: np.ndarray, cam: np.ndarray) -> str:
    """Base64 PNG: the X-ray in grey with the activation map in orange."""
    grey = ((x[0, 0] + 1024) / 2048 * 255).clip(0, 255)
    cam = resize(np.maximum(cam, 0), (224, 224), order=1, mode="edge")
    cam = cam / cam.max() if cam.max() > 0 else cam
    rgb = np.stack([grey + cam * (255 - grey), grey + cam * (140 - grey) * 0.8, grey * (1 - cam * 0.9)], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Adapter(ModelAdapter):
    id = "mlmed_torchxrayvision"

    def load(self) -> None:
        meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        onnx_path = MODEL_DIR / "model.onnx"
        if hashlib.sha256(onnx_path.read_bytes()).hexdigest() != meta["onnx_sha256"]:
            raise RuntimeError("model.onnx SHA-256 mismatch")
        self.labels = meta["labels"]
        self.thresholds = np.array(meta["op_threshs"], dtype=np.float32)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.heads = {name: json.loads((MODEL_DIR / file).read_text()) for name, file in {**ADULT_HEADS, **CHILD_HEADS}.items()}

    def head_score(self, name: str, features: np.ndarray) -> float:
        """Trained pneumonia layer, scaled like the findings: its validation threshold maps to 0.5."""
        h = self.heads[name]
        z = (features - np.array(h["mean"])) / np.array(h["scale"])
        p = float(1 / (1 + np.exp(-(z @ np.array(h["coef"]) + h["intercept"]))))
        return float(op_norm(np.array([[p]]), np.array([h["threshold"]]))[0, 0])

    def validate(self, inputs: dict) -> list[str]:
        return [] if inputs.get("image_b64") else ["image_b64: missing"]

    def scores(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, cams = self.session.run(["logits", "cams"], {"image": x})
        return op_norm(1 / (1 + np.exp(-logits)), self.thresholds), cams

    def predict(self, inputs: dict) -> Prediction:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(inputs["image_b64"])))
            img.load()
        except (ValueError, UnidentifiedImageError, OSError) as e:
            raise ValueError("the image could not be read") from e
        errors, warnings = check_image(img)
        if errors:
            raise ValueError("; ".join(errors))
        x = preprocess(img)
        scores, cams = self.scores(x)
        order = [i for i in np.argsort(-scores[0]) if self.labels[i]]
        top = order[:TOP_N]
        name = lambda i: READABLE.get(self.labels[i], self.labels[i])  # noqa: E731
        flagged = [i for i in top if scores[0, i] >= FLAG_AT]
        probabilities = {name(i): round(float(scores[0, i]), 3) for i in top}
        explanation = {"flagged": [name(i) for i in flagged], "heatmap_finding": name(top[0])}

        age = _number(inputs.get("age"))
        child = age is not None and age < CHILD_AGE
        features = self.session.run(["features"], {"image": x})[0][0]
        checks = {name: round(self.head_score(name, features), 3) for name in (CHILD_HEADS if child else ADULT_HEADS)}
        probabilities = {**checks, **probabilities}
        explanation["group"] = "child" if child else "adult"
        if child:
            label = "Possible pneumonia signs" if checks["Pneumonia signs"] >= FLAG_AT else "No clear pneumonia signs"
        elif checks["Lung opacity signs"] >= FLAG_AT:
            label = "Possible lung opacity signs"
        elif checks["Abnormal X-ray"] >= FLAG_AT:
            label = "Something may be abnormal"
        else:
            label = "No clear abnormality found"
        explanation["flagged"] = [k for k, v in checks.items() if v >= FLAG_AT] + explanation["flagged"]
        explanation["heatmap_png_b64"] = heatmap_png(x, cams[0, top[0]])
        return Prediction(model_id=self.id, label=label, warnings=warnings, probabilities=probabilities, explanation=explanation)
