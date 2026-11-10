"""Diabetes symptom check: yes/no symptoms. Trained by the Curanex team (training/train_diabetes_symptoms.py) on the UCI
Early stage diabetes risk prediction data (Sylhet Diabetes Hospital, Bangladesh). Plain numbers: nothing to unpickle.
"""

import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction, check_schema

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "diabetes_symptom_check"
YES_NO = ["Yes", "No"]
# form key -> (training column, plain name)
SYMPTOMS = {
    "passing_urine_often": ("Polyuria", "Passing urine often"),
    "very_thirsty": ("Polydipsia", "Very thirsty"),
    "sudden_weight_loss": ("sudden weight loss", "Sudden weight loss"),
    "weakness": ("weakness", "Weakness"),
    "very_hungry": ("Polyphagia", "Very hungry"),
    "genital_thrush": ("Genital thrush", "Genital yeast infection"),
    "blurred_vision": ("visual blurring", "Blurred vision"),
    "itching": ("Itching", "Itching"),
    "irritability": ("Irritability", "Irritability"),
    "slow_healing": ("delayed healing", "Wounds heal slowly"),
    "muscle_weakness": ("partial paresis", "Weakness in a group of muscles"),
    "muscle_stiffness": ("muscle stiffness", "Muscle stiffness"),
    "hair_loss": ("Alopecia", "Hair loss"),
    "obesity": ("Obesity", "Overweight"),
}


class Adapter(ModelAdapter):
    id = "diabetes_symptom_check"
    input_schema = {
        "type": "object",
        "required": ["age", "sex", "passing_urine_often", "very_thirsty"],
        "properties": {"age": {"type": "integer", "minimum": 16, "maximum": 100}, "sex": {"type": "string"}}
        | {k: {"type": "string", "enum": YES_NO} for k in SYMPTOMS},
    }

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))

    def validate(self, inputs: dict) -> list[str]:
        return check_schema(self.input_schema, inputs)

    def predict(self, inputs: dict) -> Prediction:
        m = self.meta
        sex = str(inputs.get("sex") or "").strip().lower()[:1]
        v = {"age": float(inputs["age"]), "male": float(sex == "m")}
        unanswered = 0
        for key, (col, _) in SYMPTOMS.items():
            v[col] = float(inputs.get(key) == "Yes")
            unanswered += inputs.get(key) not in YES_NO
        x = np.array([v[f] for f in m["features"]])
        contrib = (x - np.array(m["mean"])) / np.array(m["scale"]) * np.array(m["coef"])
        p = float(1 / (1 + np.exp(-(contrib.sum() + m["intercept"]))))
        t = m["threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))
        names = {"age": "Age", "male": "Sex"} | {col: name for col, name in SYMPTOMS.values()}
        order = [i for i in np.argsort(-np.abs(contrib)) if contrib[i] != 0][:3]
        warnings = ["A blood sugar test (fasting glucose or HbA1c) is the only way to know. This check only looks at symptoms."]
        if unanswered:
            warnings.append("Symptoms you didn't answer were taken as 'no'.")
        return Prediction(
            model_id=self.id,
            label="Symptoms fit diabetes" if p >= t else "Symptoms don't clearly fit diabetes",
            warnings=warnings,
            probabilities={"Unlikely": round(1 - shown, 3), "Likely": round(shown, 3)},
            explanation={"top_factors": [[names[m["features"][i]], round(float(contrib[i]), 2)] for i in order]},
        )
