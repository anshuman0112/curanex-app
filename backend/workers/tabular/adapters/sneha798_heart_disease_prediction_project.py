"""Heart disease risk from questions a patient can answer. Trained by the Curanex team (training/train_heart.py) using the
approach of Sneha798/Heart-Disease-Prediction-Project, on UCI Heart Disease data. Plain numbers: nothing to unpickle.
"""
import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction, check_schema

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sneha798_heart_disease_prediction_project"
PAIN = {
    "Pressure on effort, eases with rest": "pain_typical",
    "Unusual chest discomfort": "pain_atypical",
    "Chest pain that doesn't feel heart related": "pain_other",
    "No chest pain": "pain_none",
}
YES_NO = ["Yes", "No"]
NAMES = {
    "age": "Age", "male": "Sex", "pain_typical": "Chest pressure on effort", "pain_atypical": "Unusual chest discomfort",
    "pain_other": "Other chest pain",
    "pain_none": "No chest pain", "resting_bp": "Blood pressure", "cholesterol": "Cholesterol", "high_fasting_sugar": "High fasting sugar",
    "pain_on_exertion": "Pain with exercise",
}
EMERGENCY = "If you have severe chest pain right now, or pain with sweating or breathlessness, don't use this check. Call 112."


class Adapter(ModelAdapter):
    id = "sneha798_heart_disease_prediction_project"
    input_schema = {
        "type": "object",
        "required": ["age", "sex", "chest_pain", "pain_on_exertion"],
        "properties": {
            "age": {"type": "integer", "minimum": 18, "maximum": 100},
            "sex": {"type": "string"},
            "chest_pain": {"type": "string", "enum": list(PAIN)},
            "pain_on_exertion": {"type": "string", "enum": YES_NO},
            "resting_bp": {"type": "number", "minimum": 70, "maximum": 250},
            "cholesterol": {"type": "number", "minimum": 80, "maximum": 700},
            "high_fasting_sugar": {"type": "string", "enum": YES_NO},
        },
    }

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / "model_meta.json").read_text())

    def validate(self, inputs: dict) -> list[str]:
        return check_schema(self.input_schema, inputs)

    def predict(self, inputs: dict) -> Prediction:
        m = self.meta
        sex = str(inputs.get("sex") or "").strip().lower()
        known = {"age": float(inputs["age"]), PAIN[inputs["chest_pain"]]: 1.0}
        known["pain_on_exertion"] = float(inputs["pain_on_exertion"] == "Yes")
        known.update({k: 0.0 for k in PAIN.values() if k not in known})
        if sex[:1] in ("m", "f"):
            known["male"] = float(sex[0] == "m")
        if inputs.get("resting_bp") not in (None, ""):
            known["resting_bp"] = float(inputs["resting_bp"])
        if inputs.get("cholesterol") not in (None, ""):
            known["cholesterol"] = float(inputs["cholesterol"])
        if inputs.get("high_fasting_sugar") in YES_NO:
            known["high_fasting_sugar"] = float(inputs["high_fasting_sugar"] == "Yes")
        x = np.array([known.get(f, md) for f, md in zip(m["features"], m["median"], strict=True)])
        contrib = (x - np.array(m["mean"])) / np.array(m["scale"]) * np.array(m["coef"])
        d = float(contrib.sum() + m["intercept"])
        a, b = m["platt"]
        p = float(1 / (1 + np.exp(-(a * d + b))))
        t = m["threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))  # scaled so 50% is the flagging point
        warnings = [EMERGENCY]
        if "resting_bp" not in known or "cholesterol" not in known:
            warnings.append("Blood pressure or cholesterol wasn't given, so typical values were used.")
        order = [i for i in np.argsort(-np.abs(contrib)) if m["features"][i] in known][:3]
        return Prediction(model_id=self.id, label="Higher risk" if p >= t else "Lower risk", warnings=warnings,
                          probabilities={"Lower risk": round(1 - shown, 3), "Higher risk": round(shown, 3)},
                          explanation={"top_factors": [[NAMES[m["features"][i]], round(float(contrib[i]), 2)] for i in order]})
