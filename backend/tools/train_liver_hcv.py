"""Liver check trained by the Curanex team (standard recipe, TRAINING_PLAN.md). The repo's gate model failed an external
test (AUC 0.32). Tabular image, no network:

  docker run --rm --network none -v <backend>/data/eval/hcv:/hcv:ro -v <models>/yahyazuher_ai_liver_disease_diagnosis_system:/out \
      -v <backend>/tools:/tools:ro backend-worker-tabular python /tools/train_liver_hcv.py

UCI HCV data: blood donors vs hepatitis, fibrosis, cirrhosis ("suspect" donors removed). Features on a normal liver
function report only. 30% held out (stratified, seed 42). Logistic regression; threshold from 5-fold out-of-fold
predictions on the training rows (highest with sensitivity >= 0.90); test scored once.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("/out")
LOG = ["ALP", "ALT", "AST", "BIL", "GGT"]
COLS = ["Age", "male", "ALB", "PROT", *LOG]


def report(y, s, t):
    p = s >= t
    y = np.asarray(y)
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(float(roc_auc_score(y, s)), 3),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def main() -> None:
    d = pd.read_csv(next(Path("/hcv").glob("*.csv")), index_col=0)
    d = d[~d["Category"].str.startswith("0s")].copy()
    y = (~d["Category"].str.startswith("0")).astype(int).values
    d["male"] = (d["Sex"] == "m").astype(float)
    X = d[COLS].astype(float)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42, stratify=y)
    medians = X.iloc[tr].median()
    Z = X.fillna(medians).copy()
    Z[LOG] = np.log1p(Z[LOG])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))
    oof = cross_val_predict(pipe, Z.iloc[tr], y[tr], cv=5, method="predict_proba")[:, 1]
    threshold = float(max(t for t in np.unique(oof) if (oof[y[tr] == 1] >= t).mean() >= 0.90))
    pipe.fit(Z.iloc[tr], y[tr])
    s = pipe.predict_proba(Z.iloc[te])[:, 1]
    metrics = {"threshold": round(threshold, 4), "test": report(y[te], s, threshold), "test_at_0_5": report(y[te], s, 0.5),
               "training_folds": report(y[tr], oof, threshold)}
    scaler, clf = pipe[0], pipe[1]
    (OUT / "model_meta.json").write_text(json.dumps({
        "columns": COLS, "log": LOG, "medians": medians.tolist(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "threshold": threshold, "metrics": metrics}, indent=1))
    print(json.dumps({k: metrics[k] for k in ("threshold", "test", "test_at_0_5")}, indent=1))
    print(dict(zip(COLS, np.round(clf.coef_[0], 2), strict=True)))


if __name__ == "__main__":
    main()
