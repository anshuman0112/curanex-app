"""WORKFLOW.md A4-A8 for Moizah1/Diabetes-Prediction. Sandbox only:

  docker run --rm --network none -e PYTHONPATH=/app -v <backend>/workers:/app/workers \n      -v <vendor>/moizah1_diabetes_prediction:/repo:ro -v <models>/moizah1_diabetes_prediction:/out \
      -v <backend>/tools:/tools:ro backend-worker-tabular python /tools/convert_moizah_diabetes.py

Loads the repo's pickle (SHA-256 pinned), rebuilds its held-out split (test_size 0.2, stratified, seed 42), picks a
screening threshold from 5-fold out-of-fold predictions on the TRAINING rows only (highest with sensitivity >= 0.90),
exports plain numbers (skl2onnx cannot convert the nested KNN imputers), checks them against the original on the
test rows and measures the test split once.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split

from workers.tabular.adapters.moizah1_diabetes_prediction import load_params, predict_proba

REPO = Path("/repo")
OUT = Path("/out")
SHA256 = "0d07c11dd0e398677065931553e040a1b81e3235fc385d1ef43ddd5445e72353"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def report(y, s, threshold):
    p = s >= threshold
    y = np.asarray(y)
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(float(roc_auc_score(y, s)), 3),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def main() -> None:
    path = REPO / "models" / "final_model.joblib"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256, "SHA-256 mismatch"
    model = joblib.load(path)
    print("loaded", type(model).__name__, flush=True)

    df = pd.read_csv(REPO / "data" / "diabetes_raw.csv", header=None, names=[*FEATURES, "Outcome"])
    for c in ZERO_AS_MISSING:
        df[c] = df[c].replace(0, np.nan)
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df["Outcome"], test_size=0.2, stratify=df["Outcome"], random_state=42)
    original = model.predict_proba(X_test)[:, 1]

    oof = cross_val_predict(clone(model), X_train, y_train, cv=5, method="predict_proba")[:, 1]
    threshold = float(max(t for t in np.unique(oof) if (oof[y_train.values == 1] >= t).mean() >= 0.90))

    folds = []
    for cc in model.calibrated_classifiers_:
        pipe = cc.estimator
        imputer, scaler, clf = pipe.named_steps["imputer"], pipe.named_steps["scaler"], pipe.named_steps["model"]
        iso = cc.calibrators[0]
        folds.append({"fit_X": np.where(np.isnan(imputer._fit_X), None, imputer._fit_X).tolist(),
                      "column_means": np.nanmean(imputer._fit_X, axis=0).tolist(), "mean": scaler.mean_.tolist(),
                      "scale": scaler.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                      "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist()})
    params_path = OUT / "model_params.json"
    params_path.write_text(json.dumps({"folds": folds}))
    params = load_params(params_path)
    ours = np.array([predict_proba(params, row) for row in X_test.values.astype(float)])
    worst = float(np.abs(ours - original).max())
    print(f"max abs diff plain numbers vs original on {len(X_test)} rows: {worst:.2e}", flush=True)
    assert worst < 1e-4, "exported numbers differ from the original"

    pipe = joblib.load(REPO / "models" / "best_pipeline_uncalibrated.joblib")
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["model"]
    meta = {
        "features": FEATURES, "threshold": threshold, "params_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
        "source_sha256": SHA256, "max_abs_diff": worst, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "train_median": X_train.median().tolist(),
        "metrics": {"threshold_from_training_folds": report(y_test, ours, threshold), "default_0_5": report(y_test, ours, 0.5)},
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"threshold": round(threshold, 4), **meta["metrics"]}, indent=1))


if __name__ == "__main__":
    main()
