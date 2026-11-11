"""Shared adapter for the survey risk screeners (training/train_brfss.py). One small subclass per condition.

Logistic models are plain numbers; boosting models are ONNX files whose SHA-256 is checked on load. Nothing is unpickled.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction, check_schema

MODEL_DIR = Path(__file__).resolve().parent / "models" / "survey"
YES_NO = ["Yes", "No"]
HEALTH = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
SMOKING = ["Never smoked", "Used to smoke", "Smoke now"]
CONDITION_LABEL = {
    "diabetes": "diabetes",
    "heart_disease": "heart disease or a heart attack",
    "stroke": "a stroke",
    "asthma": "asthma",
    "copd": "COPD, emphysema or chronic bronchitis",
    "kidney_disease": "kidney disease",
    "arthritis": "arthritis",
    "depression": "depression",
}
NAMES = {
    "age": "Age",
    "male": "Sex",
    "bmi": "BMI",
    "smoker_current": "Smoking now",
    "smoker_former": "Smoked before",
    "smokeless_tobacco": "Gutka, khaini or other chewed tobacco",
    "heavy_drinker": "Heavy drinking",
    "exercise": "Exercise",
    "general_health": "General health",
    "poor_physical_days": "Days of poor physical health",
    "poor_mental_days": "Days of poor mental health",
    "difficulty_walking": "Difficulty walking",
    "difficulty_concentrating": "Difficulty concentrating",
} | {k: v[0].upper() + v[1:] for k, v in CONDITION_LABEL.items()}
BASE_SCHEMA = {
    "age": {"type": "integer", "minimum": 18, "maximum": 100},
    "sex": {"type": "string"},
    "height_cm": {"type": "number", "minimum": 100, "maximum": 230},
    "weight_kg": {"type": "number", "minimum": 25, "maximum": 300},
    "smoking": {"type": "string", "enum": SMOKING},
    "chewed_tobacco": {"type": "string", "enum": YES_NO},
    "heavy_drinking": {"type": "string", "enum": YES_NO},
    "exercise": {"type": "string", "enum": YES_NO},
    "general_health": {"type": "string", "enum": list(HEALTH)},
    "poor_physical_days": {"type": "number", "minimum": 0, "maximum": 30},
    "poor_mental_days": {"type": "number", "minimum": 0, "maximum": 30},
    "difficulty_walking": {"type": "string", "enum": YES_NO},
    "difficulty_concentrating": {"type": "string", "enum": YES_NO},
}
REQUIRED = ["age", "sex", "smoking", "general_health"]


def schema_for(target: str) -> dict:
    props = dict(BASE_SCHEMA)
    if target == "memory_problems":  # that question is what the model predicts
        props.pop("difficulty_concentrating")
    for c in CONDITION_LABEL:
        if c != target:
            props[f"has_{c}"] = {"type": "string", "enum": YES_NO}
    return {"type": "object", "required": REQUIRED, "properties": props}


class SurveyAdapter(ModelAdapter):
    target = ""

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / f"{self.target}.json").read_text(encoding="utf-8"))
        self.session = None
        if self.meta["model"] == "boosting" and "trees" in self.meta:  # boosted trees stored as plain numbers, walked in numpy
            self.trees = [{k: np.array(v) for k, v in t.items()} for t in self.meta["trees"]]
        elif self.meta["model"] == "boosting":
            import onnxruntime as ort

            path = MODEL_DIR / f"{self.target}.onnx"
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.meta["onnx_sha256"]:
                raise RuntimeError(f"{path.name} does not match its pinned SHA-256")
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def validate(self, inputs: dict) -> list[str]:
        return check_schema(self.input_schema, inputs)

    def _row(self, inputs: dict) -> tuple[dict, list[str]]:
        notes = []
        sex = str(inputs.get("sex") or "").strip().lower()[:1]
        v = {"age": float(inputs["age"]), "general_health": float(HEALTH[inputs["general_health"]])}
        if sex in ("m", "f"):
            v["male"] = float(sex == "m")
        s = inputs["smoking"]
        v["smoker_current"], v["smoker_former"] = float(s == "Smoke now"), float(s == "Used to smoke")
        if inputs.get("height_cm") and inputs.get("weight_kg"):
            v["bmi"] = float(inputs["weight_kg"]) / (float(inputs["height_cm"]) / 100) ** 2
        for key, feat in (
            ("chewed_tobacco", "smokeless_tobacco"),
            ("heavy_drinking", "heavy_drinker"),
            ("exercise", "exercise"),
            ("difficulty_walking", "difficulty_walking"),
            ("difficulty_concentrating", "difficulty_concentrating"),
        ):
            if inputs.get(key) in YES_NO:
                v[feat] = float(inputs[key] == "Yes")
        for key in ("poor_physical_days", "poor_mental_days"):
            if inputs.get(key) not in (None, ""):
                v[key] = float(inputs[key])
        missing = []
        for c in CONDITION_LABEL:
            if f"has_{c}" in self.input_schema["properties"]:
                ans = inputs.get(f"has_{c}")
                v[c] = float(ans == "Yes") if ans in YES_NO else 0.0
                if ans not in YES_NO:
                    missing.append(CONDITION_LABEL[c])
        if missing:
            notes.append("Conditions you didn't answer were taken as 'no'.")
        if "bmi" not in v:
            notes.append("Height or weight wasn't given, so a typical BMI was used.")
        return v, notes

    def predict(self, inputs: dict) -> Prediction:
        m = self.meta
        v, notes = self._row(inputs)
        x = np.array([v.get(f, md) for f, md in zip(m["features"], m["median"], strict=True)], dtype=np.float32)
        if "trees" in m:
            raw = m["baseline"]
            for tr in self.trees:
                n = 0
                while not tr["is_leaf"][n]:
                    val = x[tr["feature_idx"][n]]
                    left = bool(tr["missing_go_to_left"][n]) if np.isnan(val) else val <= tr["num_threshold"][n]
                    n = int(tr["left"][n] if left else tr["right"][n])
                raw += float(tr["value"][n])
            p = float(1 / (1 + np.exp(-raw)))
            factors = []
        elif self.session is None:
            contrib = (x - np.array(m["mean"])) / np.array(m["scale"]) * np.array(m["coef"])
            p = float(1 / (1 + np.exp(-(contrib.sum() + m["intercept"]))))
            order = [i for i in np.argsort(-np.abs(contrib)) if m["features"][i] in v][:3]
            factors = [[NAMES[m["features"][i]], round(float(contrib[i]), 2)] for i in order]
        else:
            p = float(self.session.run(None, {self.session.get_inputs()[0].name: x[None]})[1][0][1])
            factors = []
        t = m["threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))  # scaled so 50% is the flagging point
        return Prediction(
            model_id=self.id,
            label="Higher risk" if p >= t else "Lower risk",
            warnings=notes,
            probabilities={"Lower risk": round(1 - shown, 3), "Higher risk": round(shown, 3)},
            explanation={"top_factors": factors} if factors else {},
        )
