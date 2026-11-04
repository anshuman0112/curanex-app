"""Train a pneumonia layer on the X-ray model's deep features (standard recipe, TRAINING_PLAN.md). Tabular image:

  docker run --rm --network none -v <backend>/data/eval:/data -v <backend>/workers/torch/models/mlmed_torchxrayvision:/out \
      -v <backend>/tools:/tools:ro backend-worker-tabular python /tools/train_xrv_pneumonia_head.py

Kermany train split only: 80/20 train/validation grouped by patient. C and the threshold (highest with validation
sensitivity >= 0.90) come from validation. The Kermany test split and the adult set are scored once, at the end.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

DATA = Path("/data")
OUT = Path("/out")


def load(name: str):
    d = np.load(DATA / f"features_{name}.npz", allow_pickle=False)
    return d["X"], d["scores"], d["y"], d["group"]


def report(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    p = s >= threshold
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(float(roc_auc_score(y, s)), 3),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def main() -> None:
    X, _, y, groups = load("train")
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
    assert not set(groups[tr]) & set(groups[va])
    scaler = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(scaler.transform(X[tr]), y[tr])
        auc = roc_auc_score(y[va], m.predict_proba(scaler.transform(X[va]))[:, 1])
        print(f"C={c} validation AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, c, m)
    _, c, model = best
    s_va = model.predict_proba(scaler.transform(X[va]))[:, 1]
    threshold = float(max(t for t in np.unique(s_va) if (s_va[y[va] == 1] >= t).mean() >= 0.90))
    print(f"chosen C={c} threshold={threshold:.4f} validation: {report(y[va], s_va, threshold)}", flush=True)

    def score(Z):
        return model.predict_proba(scaler.transform(Z))[:, 1]

    Xt, st, yt, _ = load("test")
    Xa, sa, ya, _ = load("adult")
    results = {
        "chosen_on_validation": {"C": c, "threshold": round(threshold, 4)},
        "kermany_test_child_head": report(yt, score(Xt), threshold),
        "adult_child_head": report(ya, score(Xa), threshold),
    }
    print(json.dumps(results, indent=1))
    (OUT / "pneumonia_child_head.json").write_text(json.dumps({
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]), "threshold": threshold, "metrics": results}))


if __name__ == "__main__":
    main()
