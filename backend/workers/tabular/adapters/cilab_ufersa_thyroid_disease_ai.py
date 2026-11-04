"""Underactive thyroid pattern from a TSH result. Trained by the Curanex team (training/train_thyroid.py) using the method of
cilab-ufersa/thyroid_disease_AI, on UCI Garavan thyroid data. Plain numbers: nothing to unpickle.

T3 and T4 are not asked for: this data uses nmol/L, while most Indian reports print ng/dL and ug/dL, and a mix-up would be silent.
The overactive model missed the sensitivity bar (0.60), so it isn't used; a very low TSH gets a plain note instead.
"""
import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction, check_schema

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "cilab_ufersa_thyroid_disease_ai"
LOW_TSH = 0.1


class Adapter(ModelAdapter):
    id = "cilab_ufersa_thyroid_disease_ai"
    input_schema = {
        "type": "object",
        "required": ["tsh"],
        "properties": {
            "tsh": {"type": "number", "minimum": 0.001, "maximum": 500},
            "age": {"type": "integer", "minimum": 1, "maximum": 100},
            "sex": {"type": "string"},
            "on_thyroxine": {"type": "string", "enum": ["Yes", "No"]},
        },
    }

    def load(self) -> None:
        meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        self.features = meta["features"]
        self.m = meta["models"]["underactive"]

    def validate(self, inputs: dict) -> list[str]:
        return check_schema(self.input_schema, inputs)

    def predict(self, inputs: dict) -> Prediction:
        m = self.m
        tsh = float(inputs["tsh"])
        sex = str(inputs.get("sex") or "").strip().lower()
        known = {"log_tsh": float(np.log1p(tsh)), "on_thyroxine": float(inputs.get("on_thyroxine") == "Yes"), "on_antithyroid": 0.0}
        if inputs.get("age") not in (None, ""):
            known["age"] = float(inputs["age"])
        if sex[:1] in ("m", "f"):
            known["female"] = float(sex[0] == "f")
        x = np.array([known.get(f, md) for f, md in zip(self.features, m["median"], strict=True)])
        d = float(((x - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"]) + m["intercept"])
        p = float(1 / (1 + np.exp(-d)))
        t = m["threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))  # scaled so 50% is the flagging point
        warnings = []
        if tsh < LOW_TSH:
            warnings.append("Your TSH is very low. That can point to an overactive thyroid, which this check can't assess. "
                            "Please show this result to a doctor.")
        if inputs.get("on_thyroxine") == "Yes":
            warnings.append("You take thyroid medicine, so please ask your doctor what this TSH means for your dose.")
        flagged = p >= t
        label = "Possible underactive thyroid" if flagged else "No clear underactive thyroid pattern"
        return Prediction(model_id=self.id, label=label, warnings=warnings,
                          probabilities={"Underactive pattern": round(shown, 3), "No clear pattern": round(1 - shown, 3)})
