"""Shared adapter for the NHANES lab-checked screeners (training/train_nhanes.py). One small subclass per model.

Logistic models are plain numbers; nothing is unpickled. Values outside what the model was trained on are refused.
"""

import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction, check_schema

MODEL_DIR = Path(__file__).resolve().parent / "models" / "nhanes"
YES_NO = ["Yes", "No"]
FIELDS = {
    "age": ({"type": "integer", "minimum": 12, "maximum": 100}, "Age", None),
    "sex": ({"type": "string"}, "Sex", None),
    "height_cm": ({"type": "number", "minimum": 100, "maximum": 230}, "Height", "cm"),
    "weight_kg": ({"type": "number", "minimum": 25, "maximum": 300}, "Weight", "kg"),
    "waist_cm": ({"type": "number", "minimum": 40, "maximum": 200}, "Waist (around the belly button)", "cm"),
    "alt": ({"type": "number", "minimum": 1, "maximum": 1000}, "ALT (SGPT)", "U/L"),
    "ast": ({"type": "number", "minimum": 1, "maximum": 1000}, "AST (SGOT)", "U/L"),
    "ggt": ({"type": "number", "minimum": 1, "maximum": 2000}, "GGT", "U/L"),
    "triglycerides": ({"type": "number", "minimum": 10, "maximum": 3000}, "Triglycerides", "mg/dL"),
    "hba1c": ({"type": "number", "minimum": 3, "maximum": 18}, "HbA1c", "%"),
    "diabetes": ({"type": "string", "enum": YES_NO}, "Has a doctor told you that you have diabetes?", None),
    "hemoglobin": ({"type": "number", "minimum": 4, "maximum": 22}, "Haemoglobin (Hb)", "g/dL"),
    "mcv": ({"type": "number", "minimum": 40, "maximum": 130}, "MCV", "fL"),
    "mch": ({"type": "number", "minimum": 10, "maximum": 45}, "MCH", "pg"),
    "rdw": ({"type": "number", "minimum": 8, "maximum": 35}, "RDW", "%"),
    "platelets": ({"type": "number", "minimum": 10, "maximum": 1500}, "Platelets", "thousand/µL"),
}
PROFILE = {"age", "sex", "height_cm", "weight_kg"}
NAMES = {
    "age": "Age",
    "male": "Sex",
    "bmi": "BMI",
    "waist_cm": "Waist",
    "alt": "ALT",
    "ast": "AST",
    "ggt": "GGT",
    "triglycerides": "Triglycerides",
    "hba1c": "HbA1c",
    "diabetes": "Diabetes",
    "hemoglobin": "Haemoglobin",
    "mcv": "MCV",
    "mch": "MCH",
    "rdw": "RDW",
    "platelets": "Platelets",
}
# per model: form fields in order, required fields
FORMS = {
    "fatty_liver": (
        ["age", "sex", "alt", "ast", "waist_cm", "ggt", "triglycerides", "hba1c", "diabetes", "height_cm", "weight_kg"],
        ["age", "sex", "alt", "ast"],
    ),
    "iron_deficiency": (["age", "sex", "hemoglobin", "mcv", "rdw", "mch", "platelets"], ["age", "sex", "hemoglobin", "mcv"]),
}


def schema_for(target: str, labels: bool = False) -> dict:
    keys, required = FORMS[target]
    props = {}
    for k in keys:
        spec, label, unit = FIELDS[k]
        spec = dict(spec)
        if labels:
            spec["label"] = label
            if unit:
                spec["unit"] = unit
            if k in PROFILE:
                spec["from_profile"] = True
        props[k] = spec
    return {"type": "object", "required": required, "properties": props}


class NhanesAdapter(ModelAdapter):
    target = ""
    high = "Higher risk"
    low = "Lower risk"
    note = ""

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / f"{self.target}.json").read_text(encoding="utf-8"))

    def validate(self, inputs: dict) -> list[str]:
        errors = check_schema(self.input_schema, inputs)
        if not errors and float(inputs["age"]) < self.meta["min_age"]:
            errors.append(f"age: this check is for people aged {self.meta['min_age']} and over")
        return errors

    def predict(self, inputs: dict) -> Prediction:
        m = self.meta
        sex = str(inputs.get("sex") or "").strip().lower()[:1]
        v = {k: float(inputs[k]) for k in FIELDS if k not in ("sex", "diabetes") and inputs.get(k) not in (None, "")}
        if sex in ("m", "f"):
            v["male"] = float(sex == "m")
        if inputs.get("diabetes") in YES_NO:
            v["diabetes"] = float(inputs["diabetes"] == "Yes")
        if "height_cm" in v and "weight_kg" in v:
            v["bmi"] = v["weight_kg"] / (v["height_cm"] / 100) ** 2
        x = np.array([v.get(f, np.nan) for f in m["features"]], dtype=float)
        missing = [NAMES[f] for f, val in zip(m["features"], x, strict=True) if np.isnan(val)]
        x = np.where(np.isnan(x), np.array(m["impute"]), x)
        contrib = (x - np.array(m["mean"])) / np.array(m["scale"]) * np.array(m["coef"])
        p = float(1 / (1 + np.exp(-(contrib.sum() + m["intercept"]))))
        t = m["threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))
        warnings = [self.note] if self.note else []
        if missing:
            warnings.append(f"Not given, so typical values were used: {', '.join(missing)}.")
        order = [i for i in np.argsort(-np.abs(contrib)) if m["features"][i] in v][:3]
        return Prediction(
            model_id=self.id,
            label=self.high if p >= t else self.low,
            warnings=warnings,
            probabilities={self.low: round(1 - shown, 3), self.high: round(shown, 3)},
            explanation={"top_factors": [[NAMES[m["features"][i]], round(float(contrib[i]), 2)] for i in order]},
        )
