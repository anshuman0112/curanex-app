"""Liver function report check. Logistic model trained by the Curanex team on UCI HCV data, using the idea of a first-pass
liver gate from yahyazuher/AI-Liver-Disease-Diagnosis-System (tools/train_liver_hcv.py). Plain numbers, no pickle.
"""
import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "yahyazuher_ai_liver_disease_diagnosis_system"
NAMES = {"Age": "Age", "male": "Sex", "ALB": "Albumin", "PROT": "Total protein", "ALP": "ALP", "ALT": "ALT", "AST": "AST",
         "BIL": "Bilirubin", "GGT": "GGT"}


class Adapter(ModelAdapter):
    id = "yahyazuher_ai_liver_disease_diagnosis_system"
    input_schema = {
        "type": "object",
        "required": ["alt", "ast", "bilirubin", "age"],
        "properties": {
            "alt": {"type": "number", "minimum": 1, "maximum": 5000}, "ast": {"type": "number", "minimum": 1, "maximum": 5000},
            "bilirubin": {"type": "number", "minimum": 0.05, "maximum": 50}, "age": {"type": "integer", "minimum": 18, "maximum": 100},
            "alp": {"type": "number", "minimum": 5, "maximum": 3000}, "albumin": {"type": "number", "minimum": 1, "maximum": 7},
            "total_protein": {"type": "number", "minimum": 2, "maximum": 12}, "ggt": {"type": "number", "minimum": 1, "maximum": 3000},
        },
    }

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / "model_meta.json").read_text())

    def predict(self, inputs: dict) -> Prediction:
        m = self.meta

        def num(key: str, factor: float = 1.0) -> float | None:
            v = inputs.get(key)
            return float(v) * factor if v not in (None, "") else None

        sex = str(inputs.get("sex") or "").lower()
        raw = {"Age": num("age"), "male": 1.0 if sex.startswith("m") else 0.0 if sex.startswith("f") else None,
               "ALB": num("albumin", 10), "PROT": num("total_protein", 10), "ALP": num("alp"), "ALT": num("alt"), "AST": num("ast"),
               "BIL": num("bilirubin", 17.1), "GGT": num("ggt")}  # report units -> the g/L and umol/L the model was trained on
        cols = m["columns"]
        x = [raw[c] if raw[c] is not None else m["medians"][i] for i, c in enumerate(cols)]
        x = np.array([np.log1p(v) if c in m["log"] else v for v, c in zip(x, cols, strict=True)])
        contrib = (x - np.array(m["mean"])) / np.array(m["scale"]) * np.array(m["coef"])
        p = float(1 / (1 + np.exp(-(contrib.sum() + m["intercept"]))))
        order = [i for i in np.argsort(-np.abs(contrib)) if raw[cols[i]] is not None][:3]
        label = "Liver tests need a closer look" if p >= m["threshold"] else "Liver tests look less concerning"
        return Prediction(model_id=self.id, label=label,
                          probabilities={"Less concerning": round(1 - p, 3), "Needs a closer look": round(p, 3)},
                          explanation={"top_factors": [[NAMES[cols[i]], round(float(contrib[i]), 2)] for i in order]})
