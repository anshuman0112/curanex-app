"""Gestational diabetes risk: Keras network by srishti-chordia, converted to ONNX. Lab values the user doesn't have are filled
with typical values, and a separate threshold (set for sugar, age and BMI only) applies then.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from workers.common.adapter import ModelAdapter, Prediction

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "srishti_chordia_gestational_diabetes_prediction_using_dl"
MIN_LABS_FOR_FULL_ACCURACY = 8
LABS = {"alt": "ALT", "ast": "AST", "ggt": "GGT", "alp": "ALP", "creatinine": "CREA"}


class Adapter(ModelAdapter):
    id = "srishti_chordia_gestational_diabetes_prediction_using_dl"
    input_schema = {
        "type": "object",
        "required": ["fasting_glucose", "age", "bmi"],
        "properties": {
            "fasting_glucose": {"type": "number", "minimum": 40, "maximum": 400}, "age": {"type": "integer", "minimum": 14, "maximum": 60},
            "bmi": {"type": "number", "minimum": 12, "maximum": 60}, "alt": {"type": "number", "minimum": 1, "maximum": 2000},
            "ast": {"type": "number", "minimum": 1, "maximum": 2000}, "ggt": {"type": "number", "minimum": 1, "maximum": 2000},
            "alp": {"type": "number", "minimum": 5, "maximum": 2000}, "creatinine": {"type": "number", "minimum": 5, "maximum": 1000},
        },
    }

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        path = MODEL_DIR / "model.onnx"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.meta["onnx_sha256"]:
            raise RuntimeError("model.onnx SHA-256 mismatch")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def predict(self, inputs: dict) -> Prediction:
        m = self.meta
        if str(inputs.get("sex") or "").lower().startswith("m"):
            raise ValueError("this check is only for pregnancy")
        row = dict(zip(m["features"], m["median"], strict=True))
        row["FPG"] = float(inputs["fasting_glucose"]) / 18.0  # mg/dL to mmol/L
        row.update({"Age": float(inputs["age"]), "BMI": float(inputs["bmi"])})
        given = 0
        for key, col in LABS.items():
            if inputs.get(key) not in (None, ""):
                row[col] = float(inputs[key])
                given += 1
        x = (np.array([row[c] for c in m["features"]]) - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])
        p = float(self.session.run(None, {self.session.get_inputs()[0].name: x[np.newaxis].astype(np.float32)})[0].ravel()[0])
        threshold = m["threshold"] if given >= MIN_LABS_FOR_FULL_ACCURACY else m["threshold_few"]
        warnings = []
        if given < MIN_LABS_FOR_FULL_ACCURACY:
            warnings.append("Most lab values were missing, so this is a rough estimate that errs on the side of caution.")
        scaled = p / (threshold * 2) if p < threshold else 1 - (1 - p) / ((1 - threshold) * 2)  # threshold shows as 50%
        return Prediction(model_id=self.id, label="Higher risk" if p >= threshold else "Lower risk", warnings=warnings,
                          probabilities={"Lower risk": round(1 - scaled, 3), "Higher risk": round(scaled, 3)})
