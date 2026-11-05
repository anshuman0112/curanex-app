"""Maternal health risk (high vs not high). Trained by the Curanex team using the method of
rash200319/Two-stage-preeclampsia-AI-model, on its Bangladesh maternal data, without the simulated environment columns.
This is NOT a preeclampsia check: the data has no preeclampsia label (owner's decision, INTEGRATION_REPORT.md). Run from backend/.

Rules fixed before scoring: duplicates removed before splitting so no row sits on both sides; 70/15/15 split; threshold is the
highest with validation sensitivity >= 0.85; boosting only if it beats logistic by 0.02 AUC in training folds; test scored once.
Stress check: only age and blood pressure known.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_thyroid import export_trees, report, threshold_for

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "vendor" / "rash200319_two_stage_preeclampsia_ai_model" / "data" / "Dataset - Updated.csv"
OUT = BACKEND / "workers" / "tabular" / "models" / "rash200319_two_stage_preeclampsia_ai_model"
COLUMNS = ["Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI", "Previous Complications", "Preexisting Diabetes", "Gestational Diabetes", "Mental Health", "Heart Rate"]
FEATURES = ["age", "systolic", "diastolic", "sugar_mmol", "temp_f", "bmi", "previous_complications", "preexisting_diabetes", "gestational_diabetes", "mental_health", "heart_rate"]
BASIC = ["age", "systolic", "diastolic"]


def load():
    seen, rows, y = set(), [], []
    with DATA.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["Risk Level"] not in ("High", "Low"):
                continue
            v = tuple(float(r[c]) if r[c] not in ("", "NA", "nan") else np.nan for c in COLUMNS)
            key = (tuple("x" if a != a else a for a in v), r["Risk Level"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(v)
            y.append(int(r["Risk Level"] == "High"))
    return np.array(rows), np.array(y)


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def gb():
    return make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.06, random_state=42))


def main() -> None:
    x, y = load()
    print(len(y), int(y.sum()), flush=True)
    tr, rest = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42, stratify=y)
    va, te = train_test_split(rest, test_size=0.50, random_state=42, stratify=y[rest])
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    a_lr = roc_auc_score(y[tr], cross_val_predict(lr(), x[tr], y[tr], cv=folds, method="predict_proba")[:, 1])
    a_gb = roc_auc_score(y[tr], cross_val_predict(gb(), x[tr], y[tr], cv=folds, method="predict_proba")[:, 1])
    print(f"training-fold AUC: logistic {a_lr:.3f}, boosting {a_gb:.3f}", flush=True)
    make = gb if a_gb > a_lr + 0.02 else lr
    model = make().fit(x[tr], y[tr])
    t = threshold_for(y[va], model.predict_proba(x[va])[:, 1])
    basic = x[te].copy()
    basic[:, [i for i, f in enumerate(FEATURES) if f not in BASIC]] = np.nan
    metrics = {"model": make.__name__, "test": report(y[te], model.predict_proba(x[te])[:, 1], t), "only_age_and_bp": report(y[te], model.predict_proba(basic)[:, 1], t)}
    print(metrics, flush=True)
    meta = {"features": FEATURES, "training_rows": len(tr), "threshold": t, "metrics": metrics}
    if make is gb:
        meta.update({"kind": "trees", **export_trees(model)})
    else:
        imp, sc, clf = model.steps[0][1], model.steps[1][1], model.steps[2][1]
        meta.update({"kind": "logistic", "median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_meta.json").write_text(json.dumps(meta))
    print("wrote", (OUT / "model_meta.json").stat().st_size)


if __name__ == "__main__":
    main()
