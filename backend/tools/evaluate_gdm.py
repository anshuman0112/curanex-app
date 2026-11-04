"""WORKFLOW.md A8 for the gestational diabetes model. Tabular image (the scaler is a pickle), no network:

  docker run --rm --network none -v <vendor>/srishti_chordia_gestational_diabetes_prediction_using_dl:/repo:ro       -v <models>/srishti_chordia_gestational_diabetes_prediction_using_dl:/out -v <backend>/tools:/tools:ro backend-worker-tabular python /tools/evaluate_gdm.py

Rebuilds the repo's held-out split (test_size 0.2, seed 42). Half of it (seed 42) sets the threshold (highest with
sensitivity >= 0.90), the other half is scored once. Rows with long decimals look synthetic, so they are also reported apart.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path("/repo")
OUT = Path("/out")


def report(y, s, t):
    p = s >= t
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(float(roc_auc_score(y, s)), 3), "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def main() -> None:
    session = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
    df = pd.read_csv(REPO / "gestational_diabetes.csv")
    # The saved model takes all 25 columns. Its scaler was fitted on X_train in the notebook and not saved (scaler.joblib has
    # 18 columns), so it is rebuilt from the same split.
    X, y = df.drop(columns=["GDM"]), df["GDM"]
    X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    Z = scaler.transform(X_test).astype(np.float32)
    s = session.run(None, {session.get_inputs()[0].name: Z})[0].ravel()
    yt = y_test.values
    va, te = train_test_split(np.arange(len(yt)), test_size=0.5, random_state=42, stratify=yt)
    threshold = float(max(t for t in np.unique(s[va]) if (s[va][yt[va] == 1] >= t).mean() >= 0.90))
    long_dec = (df["ALT"].astype(str).str.split(".").str[-1].str.len() > 3).loc[X_test.index].values
    real = te[~long_dec[te]]
    metrics = {"threshold": round(threshold, 4), "test": report(yt[te], s[te], threshold), "test_at_0_5": report(yt[te], s[te], 0.5),
               "share_rows_long_decimals": round(float(long_dec.mean()), 3), "real_looking_rows_only": report(yt[real], s[real], threshold)}
    # What a phone user has: only fasting sugar, age and BMI known, everything else at the training median
    few = X_test.copy()
    known = ["FPG", "Age", "BMI"]
    few[[c for c in X.columns if c not in known]] = X_train.median()[[c for c in X.columns if c not in known]].values
    s_few = session.run(None, {session.get_inputs()[0].name: scaler.transform(few).astype(np.float32)})[0].ravel()
    t_few = float(max(t for t in np.unique(s_few[va]) if (s_few[va][yt[va] == 1] >= t).mean() >= 0.80))  # 0.90 flagged normal sugars
    metrics["only_sugar_age_bmi"] = {"threshold": round(t_few, 4), **report(yt[real], s_few[real], t_few)}
    conv = json.loads((OUT / "convert.json").read_text())
    (OUT / "model_meta.json").write_text(json.dumps({
        "features": X.columns.tolist(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "median": X_train.median().tolist(), "threshold": threshold, "threshold_few": t_few, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "onnx_sha256": hashlib.sha256((OUT / "model.onnx").read_bytes()).hexdigest(), **conv, "metrics": metrics}, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
