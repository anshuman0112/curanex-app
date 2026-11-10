"""Diabetes symptom check: yes/no symptoms a person can answer. Trained by the Curanex team on the UCI "Early stage
diabetes risk prediction" data (Islam et al. 2020: 520 people at Sylhet Diabetes Hospital, Bangladesh). Run from backend/.

Rules fixed before looking at the test split: 60/20/20 stratified; logistic regression (plain numbers);
threshold highest with validation sensitivity >= 0.85; test scored once. No outside set exists, so a pass makes it
live but Unverified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "eval" / "uci" / "529" / "diabetes_data_upload.csv"
OUT = BACKEND / "workers" / "tabular" / "models" / "diabetes_symptom_check" / "model_meta.json"
SYMPTOMS = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]
FEATURES = ["age", "male", *SYMPTOMS]


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "accuracy": round(float((tp + tn) / len(y)), 3),
        "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
        "specificity": round(float(tn / max(tn + fp, 1)), 3),
        "auc": round(float(roc_auc_score(y, p)), 3),
    }


def main() -> None:
    d = pd.read_csv(DATA)
    x = np.column_stack([d["Age"], (d["Gender"] == "Male")] + [(d[s] == "Yes") for s in SYMPTOMS]).astype(float)
    y = (d["class"] == "Positive").to_numpy(int)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, random_state=42, stratify=y)
    va, te = train_test_split(rest, test_size=0.5, random_state=42, stratify=y[rest])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(x[tr], y[tr])
    pv = model.predict_proba(x[va])[:, 1]
    t = float(max(v for v in np.unique(pv) if (pv[y[va] == 1] >= v).mean() >= 0.85))
    test = report(y[te], model.predict_proba(x[te])[:, 1], t)
    print("test (once):", test, "threshold", round(t, 3))
    sc, clf = model.steps[0][1], model.steps[1][1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "mean": sc.mean_.tolist(),
                "scale": sc.scale_.tolist(),
                "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]),
                "threshold": t,
                "metrics": {"test": test},
                "training_rows": len(tr),
            },
            indent=1,
        )
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
