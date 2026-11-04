"""WORKFLOW.md A4-A8 for the gate model of yahyazuher/AI-Liver-Disease-Diagnosis-System. Sandbox only:

  docker run --rm --network none -v <vendor>/yahyazuher_ai_liver_disease_diagnosis_system:/repo:ro \
      -v <models>/yahyazuher_liver_gate:/out -v <backend>/data/eval/hcv:/hcv:ro -v <backend>/tools:/tools:ro \
      backend-worker-tabular python /tools/convert_yahyazuher_liver.py

Loads the pickle (SHA-256 pinned), saves the booster as XGBoost JSON (no pickle at runtime). Its training data contains
ILPD, so ILPD can't test it. External test: UCI HCV data (615 people, Germany): blood donors vs hepatitis, fibrosis,
cirrhosis. Direct bilirubin isn't in HCV, so it's left missing; A/G ratio = ALB / (PROT - ALB). HCV is split 50/50
(seed 42): threshold from one half (highest with sensitivity >= 0.90), scored once on the other.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path("/repo")
OUT = Path("/out")
SHA256 = None  # printed on first run, then pinned in model_meta.json
FEATURES = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "ALP", "ALT", "AST", "Total_Protiens", "Albumin",
            "Albumin_and_Globulin_Ratio"]


def report(y, s, t):
    p = s >= t
    y = np.asarray(y)
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(float(roc_auc_score(y, s)), 3),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def main() -> None:
    path = REPO / "models" / "gate_model.pkl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    model = joblib.load(path)
    print("loaded", type(model).__name__, "features", getattr(model, "feature_names_in_", None), flush=True)
    booster = model.get_booster()
    booster.save_model(str(OUT / "model.json"))

    hcv = pd.read_csv(next(Path("/hcv").glob("*.csv")), index_col=0)
    hcv = hcv[~hcv["Category"].str.startswith("0s")]  # "suspect blood donor" is unclear either way
    X = pd.DataFrame({
        "Age": hcv["Age"].astype(float), "Gender": (hcv["Sex"] == "m").astype(float), "Total_Bilirubin": hcv["BIL"] * 0.0585,
        "Direct_Bilirubin": np.nan, "ALP": hcv["ALP"], "ALT": hcv["ALT"], "AST": hcv["AST"], "Total_Protiens": hcv["PROT"] / 10,
        "Albumin": hcv["ALB"] / 10, "Albumin_and_Globulin_Ratio": hcv["ALB"] / (hcv["PROT"] - hcv["ALB"])})[FEATURES]
    y = (~hcv["Category"].str.startswith("0")).astype(int)
    fresh = xgb.Booster()
    fresh.load_model(str(OUT / "model.json"))
    sick = 1 - fresh.predict(xgb.DMatrix(X.values.astype(float), feature_names=FEATURES))  # class 1 = healthy
    va, te = train_test_split(np.arange(len(y)), test_size=0.5, random_state=42, stratify=y)
    threshold = float(max(t for t in np.unique(sick[va]) if (sick[va][y.values[va] == 1] >= t).mean() >= 0.90))
    metrics = {"threshold": round(threshold, 4), "external_test_half": report(y.values[te], sick[te], threshold),
               "external_test_half_at_0_5": report(y.values[te], sick[te], 0.5)}
    (OUT / "model_meta.json").write_text(json.dumps({
        "features": FEATURES, "threshold": threshold, "source_sha256": digest,
        "model_sha256": hashlib.sha256((OUT / "model.json").read_bytes()).hexdigest(), "metrics": metrics}, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
