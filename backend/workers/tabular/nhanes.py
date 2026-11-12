"""Shared adapter for the NHANES lab-checked screeners (training/train_nhanes.py). One small subclass per model.

Logistic models are plain numbers; nothing is unpickled. Values outside what the model was trained on are refused.
"""

import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction, check_schema

MODEL_DIR = Path(__file__).resolve().parent / "models" / "nhanes"
YES_NO = ["Yes", "No"]
HEARING = {"Excellent": 1, "Good": 2, "A little trouble": 3, "Moderate trouble": 4, "A lot of trouble": 5, "Deaf": 6}
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
    "hdl": ({"type": "number", "minimum": 5, "maximum": 200}, "HDL cholesterol (good cholesterol)", "mg/dL"),
    "uric_acid": ({"type": "number", "minimum": 0.5, "maximum": 20}, "Uric acid", "mg/dL"),
    "bun": ({"type": "number", "minimum": 1, "maximum": 200}, "Urea nitrogen (BUN)", "mg/dL"),
    "albumin": ({"type": "number", "minimum": 1, "maximum": 7}, "Albumin", "g/dL"),
    "systolic_bp": ({"type": "number", "minimum": 70, "maximum": 260}, "Blood pressure, top number (systolic)", "mmHg"),
    "diabetes": ({"type": "string", "enum": YES_NO}, "Has a doctor told you that you have diabetes?", None),
    "hemoglobin": ({"type": "number", "minimum": 4, "maximum": 22}, "Haemoglobin (Hb)", "g/dL"),
    "mcv": ({"type": "number", "minimum": 40, "maximum": 130}, "MCV", "fL"),
    "mch": ({"type": "number", "minimum": 10, "maximum": 45}, "MCH", "pg"),
    "rdw": ({"type": "number", "minimum": 8, "maximum": 35}, "RDW", "%"),
    "platelets": ({"type": "number", "minimum": 10, "maximum": 1500}, "Platelets", "thousand/µL"),
    "smoking": ({"type": "string", "enum": ["Never smoked", "Used to smoke", "Smoke now"]}, "Do you smoke?", None),
    "high_bp": ({"type": "string", "enum": YES_NO}, "Has a doctor told you that you have high blood pressure?", None),
    "heart_or_stroke": ({"type": "string", "enum": YES_NO}, "Have you had heart disease, a heart attack or a stroke?", None),
    "hearing_rating": ({"type": "string", "enum": list(HEARING)}, "How is your hearing, without a hearing aid?", None),
    "hears_whisper": (
        {"type": "string", "enum": YES_NO},
        "Can you understand someone whispering across a quiet room, without seeing their face?",
        None,
    ),
    "hears_normal_voice": ({"type": "string", "enum": YES_NO}, "Can you understand a normal voice across a quiet room?", None),
    "ringing_ears": ({"type": "string", "enum": YES_NO}, "Have you had ringing or buzzing in your ears in the last year?", None),
    "heart_attack_or_chd": ({"type": "string", "enum": YES_NO}, "Have you had a heart attack or coronary heart disease?", None),
    "stroke": ({"type": "string", "enum": YES_NO}, "Have you had a stroke?", None),
}
YES_NO_FEATURES = {
    "diabetes",
    "high_bp",
    "heart_or_stroke",
    "hears_whisper",
    "hears_normal_voice",
    "ringing_ears",
    "heart_attack_or_chd",
    "stroke",
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
    "hdl": "HDL cholesterol",
    "uric_acid": "Uric acid",
    "bun": "Urea nitrogen (BUN)",
    "albumin": "Albumin",
    "systolic_bp": "Blood pressure (top number)",
    "diabetes": "Diabetes",
    "hemoglobin": "Haemoglobin",
    "mcv": "MCV",
    "mch": "MCH",
    "rdw": "RDW",
    "platelets": "Platelets",
    "weight_kg": "Weight",
    "height_cm": "Height",
    "smoker_current": "Smoking now",
    "smoker_ever": "Ever smoked",
    "high_bp": "High blood pressure",
    "heart_or_stroke": "Heart disease or stroke",
    "hearing_rating": "Your hearing rating",
    "hears_whisper": "Hearing a whisper",
    "hears_normal_voice": "Hearing a normal voice",
    "ringing_ears": "Ringing in the ears",
    "heart_attack_or_chd": "Heart attack or coronary disease",
    "stroke": "Stroke",
}
# per model: form fields in order, required fields
FORMS = {
    "fatty_liver": (
        ["age", "sex", "alt", "ast", "waist_cm", "ggt", "triglycerides", "hba1c", "diabetes", "height_cm", "weight_kg"],
        ["age", "sex", "alt", "ast"],
    ),
    "iron_deficiency": (["age", "sex", "hemoglobin", "mcv", "rdw", "mch", "platelets"], ["age", "sex", "hemoglobin", "mcv"]),
    "osteoporosis": (["age", "sex", "weight_kg", "height_cm", "smoking", "diabetes"], ["age", "sex", "weight_kg", "height_cm", "smoking"]),
    "kidney_disease_lab": (
        ["age", "sex", "diabetes", "high_bp", "heart_or_stroke", "smoking", "height_cm", "weight_kg"],
        ["age", "sex", "diabetes", "high_bp"],
    ),
    "low_bone_density": (
        ["age", "sex", "weight_kg", "height_cm", "smoking", "diabetes"],
        ["age", "sex", "weight_kg", "height_cm", "smoking"],
    ),
    "gout": (
        ["age", "sex", "high_bp", "diabetes", "waist_cm", "heart_or_stroke", "smoking", "height_cm", "weight_kg"],
        ["age", "sex", "high_bp", "diabetes"],
    ),
    "heart_failure": (
        ["age", "sex", "high_bp", "heart_attack_or_chd", "diabetes", "stroke", "smoking", "height_cm", "weight_kg"],
        ["age", "sex", "high_bp", "heart_attack_or_chd"],
    ),
    "hidden_diabetes_labs": (
        ["age", "sex", "triglycerides", "hdl", "waist_cm", "systolic_bp", "alt", "ggt", "uric_acid", "high_bp", "smoking", "height_cm", "weight_kg"],
        ["age", "sex", "triglycerides", "hdl"],
    ),
    "kidney_damage_labs": (
        ["age", "sex", "bun", "uric_acid", "systolic_bp", "high_bp", "diabetes", "triglycerides", "hdl", "height_cm", "weight_kg"],
        ["age", "sex", "bun", "uric_acid"],
    ),
    "low_egfr_labs": (
        ["age", "sex", "bun", "uric_acid", "systolic_bp", "high_bp", "diabetes", "hdl", "height_cm", "weight_kg"],
        ["age", "sex", "bun", "uric_acid"],
    ),
    "hepatitis_c_labs": (["age", "sex", "alt", "ast", "platelets", "albumin", "ggt", "smoking"], ["age", "sex", "alt", "ast", "platelets"]),
    "anemia_labs": (["age", "sex", "mcv", "mch", "rdw", "platelets"], ["age", "sex", "mcv", "mch", "rdw"]),
    "fatty_liver_questions": (["age", "sex", "waist_cm", "height_cm", "weight_kg", "diabetes", "high_bp", "smoking"], ["age", "sex", "waist_cm", "height_cm", "weight_kg"]),
    "hearing_loss": (
        ["age", "sex", "hearing_rating", "hears_whisper", "hears_normal_voice", "ringing_ears"],
        ["age", "sex", "hearing_rating", "hears_whisper"],
    ),
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
        v = {
            k: float(inputs[k])
            for k, (spec, _, _) in FIELDS.items()
            if spec["type"] in ("number", "integer") and inputs.get(k) not in (None, "")
        }
        if sex in ("m", "f"):
            v["male"] = float(sex == "m")
        for k in YES_NO_FEATURES:
            if inputs.get(k) in YES_NO:
                v[k] = float(inputs[k] == "Yes")
        if inputs.get("smoking") in FIELDS["smoking"][0]["enum"]:
            v["smoker_current"] = float(inputs["smoking"] == "Smoke now")
            v["smoker_ever"] = float(inputs["smoking"] != "Never smoked")
        if inputs.get("hearing_rating") in HEARING:
            v["hearing_rating"] = float(HEARING[inputs["hearing_rating"]])
        if "height_cm" in v and "weight_kg" in v:
            v["bmi"] = v["weight_kg"] / (v["height_cm"] / 100) ** 2
        x = np.array([v.get(f, np.nan) for f in m["features"]], dtype=float)
        missing = [NAMES[f] for f, val in zip(m["features"], x, strict=True) if np.isnan(val)]
        if "trees" in m:  # boosted trees stored as plain numbers; the trees handle missing values themselves
            raw = m["baseline"]
            for tr in m["trees"]:
                n = 0
                while not tr["is_leaf"][n]:
                    val = x[tr["feature_idx"][n]]
                    left = bool(tr["missing_go_to_left"][n]) if np.isnan(val) else val <= tr["num_threshold"][n]
                    n = int(tr["left"][n] if left else tr["right"][n])
                raw += float(tr["value"][n])
            p = float(1 / (1 + np.exp(-raw)))
            contrib = np.zeros(len(x))
        else:
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
            explanation={"top_factors": [[NAMES[m["features"][i]], round(float(contrib[i]), 2)] for i in order]} if contrib.any() else {},
        )
