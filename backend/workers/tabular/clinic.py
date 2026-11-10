"""Shared adapter for clinic-grade UCI models (training/train_uci_clinic.py): values a doctor or lab enters.

Plain-number logistic models; nothing is unpickled.
"""

import json
from pathlib import Path

import numpy as np

from workers.common.adapter import ModelAdapter, Prediction, check_schema

MODEL_DIR = Path(__file__).resolve().parent / "models" / "clinic"
SCORE = {"type": "integer", "minimum": 1, "maximum": 10}
MUTATION = {"type": "integer", "minimum": 0, "maximum": 1}
# per model: form key -> (schema, label, unit); the order is the order asked; the first five are required
FORMS = {
    "fetal_ctg": {
        "astv": ({"type": "number", "minimum": 0, "maximum": 100}, "Abnormal short-term variability (ASTV)", "%"),
        "altv": ({"type": "number", "minimum": 0, "maximum": 100}, "Abnormal long-term variability (ALTV)", "%"),
        "accelerations": ({"type": "number", "minimum": 0, "maximum": 50}, "Accelerations in the trace (AC)", "count"),
        "prolonged_decelerations": ({"type": "number", "minimum": 0, "maximum": 20}, "Prolonged decelerations (DP)", "count"),
        "baseline_fhr": ({"type": "number", "minimum": 90, "maximum": 200}, "Baseline fetal heart rate (LB)", "bpm"),
        "mstv": ({"type": "number", "minimum": 0, "maximum": 10}, "Mean short-term variability (MSTV)", "ms"),
        "contractions": ({"type": "number", "minimum": 0, "maximum": 50}, "Uterine contractions (UC)", "count"),
        "light_decelerations": ({"type": "number", "minimum": 0, "maximum": 50}, "Light decelerations (DL)", "count"),
    },
    "breast_fna": {
        "clump_thickness": (SCORE, "Clump thickness (1-10)", None),
        "cell_size_uniformity": (SCORE, "Uniformity of cell size (1-10)", None),
        "cell_shape_uniformity": (SCORE, "Uniformity of cell shape (1-10)", None),
        "bare_nuclei": (SCORE, "Bare nuclei (1-10)", None),
        "bland_chromatin": (SCORE, "Bland chromatin (1-10)", None),
        "normal_nucleoli": (SCORE, "Normal nucleoli (1-10)", None),
        "marginal_adhesion": (SCORE, "Marginal adhesion (1-10)", None),
        "mitoses": (SCORE, "Mitoses (1-10)", None),
    },
    "mammo_mass": {
        "birads": ({"type": "integer", "minimum": 0, "maximum": 6}, "BI-RADS category given by the radiologist (0-6)", None),
        "age": ({"type": "integer", "minimum": 18, "maximum": 100}, "Age", "years"),
        "shape": ({"type": "integer", "minimum": 1, "maximum": 4}, "Mass shape: 1 round, 2 oval, 3 lobular, 4 irregular", None),
        "margin": (
            {"type": "integer", "minimum": 1, "maximum": 5},
            "Mass margin: 1 circumscribed, 2 microlobulated, 3 obscured, 4 ill-defined, 5 spiculated",
            None,
        ),
        "density": ({"type": "integer", "minimum": 1, "maximum": 4}, "Mass density: 1 high, 2 iso, 3 low, 4 fat-containing", None),
    },
    "glioma_grade": {
        "idh1": (MUTATION, "IDH1 mutation (1 = mutated, 0 = not)", None),
        "age": ({"type": "number", "minimum": 1, "maximum": 100}, "Age at diagnosis", "years"),
        "atrx": (MUTATION, "ATRX mutation (1 = mutated, 0 = not)", None),
        "pten": (MUTATION, "PTEN mutation (1 = mutated, 0 = not)", None),
        "egfr": (MUTATION, "EGFR mutation (1 = mutated, 0 = not)", None),
        "tp53": (MUTATION, "TP53 mutation (1 = mutated, 0 = not)", None),
        "cic": (MUTATION, "CIC mutation (1 = mutated, 0 = not)", None),
        "nf1": (MUTATION, "NF1 mutation (1 = mutated, 0 = not)", None),
        "male": (MUTATION, "Sex (1 = male, 0 = female)", None),
    },
}


def schema_for(name: str, labels: bool = False) -> dict:
    props = {}
    for k, (spec, label, unit) in FORMS[name].items():
        spec = dict(spec)
        if labels:
            spec["label"] = label
            if unit:
                spec["unit"] = unit
        props[k] = spec
    return {"type": "object", "required": list(FORMS[name])[:5], "properties": props}


class ClinicAdapter(ModelAdapter):
    name = ""
    high = "Needs attention"
    low = "Reassuring"
    note = ""

    def load(self) -> None:
        self.meta = json.loads((MODEL_DIR / f"{self.name}.json").read_text(encoding="utf-8"))

    def validate(self, inputs: dict) -> list[str]:
        return check_schema(self.input_schema, inputs)

    def predict(self, inputs: dict) -> Prediction:
        m = self.meta
        x = np.array([float(inputs[f]) if inputs.get(f) not in (None, "") else np.nan for f in m["features"]])
        missing = [FORMS[self.name][f][1] for f, v in zip(m["features"], x, strict=True) if np.isnan(v)]
        x = np.where(np.isnan(x), np.array(m["impute"]), x)
        contrib = (x - np.array(m["mean"])) / np.array(m["scale"]) * np.array(m["coef"])
        p = float(1 / (1 + np.exp(-(contrib.sum() + m["intercept"]))))
        t = m["threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))
        warnings = [self.note] if self.note else []
        if missing:
            warnings.append(f"Not given, so typical values were used: {', '.join(missing)}.")
        order = [i for i in np.argsort(-np.abs(contrib)) if not np.isnan(float(inputs.get(m["features"][i]) or np.nan))][:3]
        return Prediction(
            model_id=self.id,
            label=self.high if p >= t else self.low,
            warnings=warnings,
            probabilities={self.low: round(1 - shown, 3), self.high: round(shown, 3)},
            explanation={"top_factors": [[FORMS[self.name][m["features"][i]][1], round(float(contrib[i]), 2)] for i in order]},
        )
