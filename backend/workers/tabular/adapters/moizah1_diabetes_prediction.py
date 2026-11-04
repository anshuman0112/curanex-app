"""Type 2 diabetes risk (Pima). Moizah1/Diabetes-Prediction's calibrated logistic model, exported to plain numbers
(model_params.json): no pickle at runtime. Unknown values are filled by the model's own KNN imputer.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction, check_schema

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "moizah1_diabetes_prediction"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
KEYS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "pedigree", "age"]
NAMES = dict(zip(KEYS, ["Pregnancies", "Blood sugar", "Blood pressure", "Skin fold", "Insulin", "BMI", "Family history score", "Age"],
                 strict=True))
ZERO_IS_MISSING = {"glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"}
K = 5


def knn_impute(x: np.ndarray, fit: np.ndarray, means: np.ndarray) -> np.ndarray:
    """sklearn KNNImputer(n_neighbors=5) with nan_euclidean distance. x: one row."""
    out = x.copy()
    both = ~np.isnan(fit) & ~np.isnan(x)
    diff = np.where(both, fit - np.nan_to_num(x), 0.0)
    present = both.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.sqrt((diff ** 2).sum(axis=1) * x.size / present)
    dist[present == 0] = np.nan
    for j in np.where(np.isnan(x))[0]:
        donors = np.where(~np.isnan(fit[:, j]) & ~np.isnan(dist))[0]
        if len(donors) == 0:
            out[j] = means[j]
            continue
        nearest = donors[np.argsort(dist[donors], kind="stable")[:K]]
        out[j] = fit[nearest, j].mean()
    return out


def predict_proba(params: dict, x: np.ndarray) -> float:
    """Average of the 5 calibrated folds for one row of FEATURES (NaN = unknown)."""
    total = 0.0
    for fold in params["folds"]:
        filled = knn_impute(x, fold["fit_X"], fold["column_means"])
        z = (filled - fold["mean"]) / fold["scale"]
        d = float(z @ fold["coef"] + fold["intercept"])
        total += float(np.interp(d, fold["iso_x"], fold["iso_y"]))
    return total / len(params["folds"])


def load_params(path: Path) -> dict:
    p = json.loads(path.read_text())
    for fold in p["folds"]:
        for k in ("fit_X", "column_means", "mean", "scale", "coef", "iso_x", "iso_y"):
            fold[k] = np.array(fold[k], dtype=float)
    return p


class Adapter(ModelAdapter):
    id = "moizah1_diabetes_prediction"
    input_schema = {
        "type": "object",
        "required": ["glucose", "bmi", "age"],
        "properties": {
            "glucose": {"type": "number", "minimum": 40, "maximum": 400},
            "bmi": {"type": "number", "minimum": 10, "maximum": 70},
            "age": {"type": "integer", "minimum": 18, "maximum": 100},
            "blood_pressure": {"type": "number", "minimum": 30, "maximum": 150},
            "pregnancies": {"type": "integer", "minimum": 0, "maximum": 20},
            "insulin": {"type": "number", "minimum": 2, "maximum": 900},
            "skin_thickness": {"type": "number", "minimum": 5, "maximum": 100},
            "pedigree": {"type": "number", "minimum": 0, "maximum": 3},
        },
    }

    def load(self) -> None:
        path = MODEL_DIR / "model_params.json"
        meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["params_sha256"]:
            raise RuntimeError("model_params.json SHA-256 mismatch")
        self.meta = meta
        self.params = load_params(path)

    def predict(self, inputs: dict) -> Prediction:
        row = []
        for key in KEYS:
            v = inputs.get(key)
            v = float(v) if v not in (None, "") else np.nan
            row.append(np.nan if key in ZERO_IS_MISSING and v == 0 else v)
        x = np.array(row, dtype=float)
        sex = str(inputs.get("sex") or "").lower()
        warnings = []
        if sex.startswith("m"):
            x[0] = 0.0
            warnings.append("This model was trained only on women, so results for men are less reliable.")
        p = predict_proba(self.params, x)
        t = self.meta["threshold"]
        higher = p >= t
        p = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))  # scaled so 50% is the flagging point and the score agrees with the label
        known = ~np.isnan(x)
        filled = np.where(known, x, np.array(self.meta["train_median"]))
        contrib = (filled - np.array(self.meta["mean"])) / np.array(self.meta["scale"]) * np.array(self.meta["coef"])
        order = [i for i in np.argsort(-np.abs(contrib)) if known[i]][:3]
        return Prediction(model_id=self.id, label="Higher risk" if higher else "Lower risk", warnings=warnings,
                          probabilities={"Lower risk": round(1 - p, 3), "Higher risk": round(p, 3)},
                          explanation={"top_factors": [[NAMES[KEYS[i]], round(float(contrib[i]), 2)] for i in order]})

    def validate(self, inputs: dict) -> list[str]:
        return check_schema(self.input_schema, inputs)
