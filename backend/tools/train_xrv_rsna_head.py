"""Adult pneumonia layer: trained on RSNA radiologist labels, tested on a different hospital system. Tabular image:

  docker run --rm --network none -v <backend>/data/eval:/data -v <backend>/workers/torch/models/mlmed_torchxrayvision:/out \
      -v <backend>/tools:/tools:ro backend-worker-tabular python /tools/train_xrv_rsna_head.py

Rules fixed before running. RSNA (one image per patient): Lung Opacity = 1, Normal or other = 0; 85/15 train/validation
(seed 42). C and the threshold (highest with validation sensitivity >= 0.90) come from validation. Tests, scored once:
the VinBigData held-out 15% (same split as train_xrv_adult_head.py), and the Kermany test split.
RSNA images were in the original model's training, so validation scores are optimistic; only the tests count.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path("/data")
OUT = Path("/out")


def report(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    p = s >= threshold
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(float(roc_auc_score(y, s)), 3),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def main() -> None:
    d = np.load(DATA / "features_rsna.npz", allow_pickle=False)
    X, y = d["X"], d["y"]
    tr, va = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(scaler.transform(X[tr]), y[tr])
        auc = roc_auc_score(y[va], m.predict_proba(scaler.transform(X[va]))[:, 1])
        print(f"C={c} validation AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, c, m)
    _, c, model = best
    score = lambda Z: model.predict_proba(scaler.transform(Z))[:, 1]  # noqa: E731
    s_va = score(X[va])
    threshold = float(max(t for t in np.unique(s_va) if (s_va[y[va] == 1] >= t).mean() >= 0.90))

    v = np.load(DATA / "features_vinbig.npz", allow_pickle=False)
    col = {n: i for i, n in enumerate(v["classes"].tolist())}
    abnormal = v["labels"].any(axis=1).astype(int)
    lung = v["labels"][:, [col["Consolidation"], col["Infiltration"], col["Lung Opacity"]]].any(axis=1).astype(int)
    strata = abnormal + lung
    _, rest = train_test_split(np.arange(len(lung)), test_size=0.30, random_state=42, stratify=strata)
    _, te = train_test_split(rest, test_size=0.50, random_state=42, stratify=strata[rest])
    k = np.load(DATA / "features_test.npz", allow_pickle=False)

    results = {
        "chosen_on_validation": {"C": c, "threshold": round(threshold, 4)},
        "rsna_validation_optimistic": report(y[va], s_va, threshold),
        "vinbig_test_adults_other_country": report(lung[te], score(v["X"][te]), threshold),
        "kermany_test_children": report(k["y"], score(k["X"]), threshold),
    }
    print(json.dumps(results, indent=1))
    (OUT / "pneumonia_adult_head.json").write_text(json.dumps({
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]), "threshold": threshold, "metrics": results}))


if __name__ == "__main__":
    main()
