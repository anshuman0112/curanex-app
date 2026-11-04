"""Adult lung-finding layer on the X-ray model's features, plus a fair measurement of the original model.
Tabular image, no network:

  docker run --rm --network none -v <backend>/data/eval:/data -v <backend>/workers/torch/models/mlmed_torchxrayvision:/out \
      -v <backend>/tools:/tools:ro backend-worker-tabular python /tools/train_xrv_adult_head.py

Two layers on every extracted image. "abnormal": any finding vs all radiologists said No finding. "lung": Consolidation,
Infiltration or Lung Opacity vs everything else, including other chest problems (a first version compared against normal
X-rays only and wrongly flagged 78% of other chest problems). Split 70/15/15 (seed 42): C and the threshold (highest with
validation sensitivity >= 0.90) come from the 15% validation part; the last 15% is scored once.
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
XRV_LABELS = json.loads((OUT / "model_meta.json").read_text())["labels"]
# VinBigData class -> matching finding(s) of the original model
SAME = {"Cardiomegaly": ["Cardiomegaly"], "Pleural effusion": ["Effusion"], "Pneumothorax": ["Pneumothorax"],
        "Atelectasis": ["Atelectasis"],
        "Consolidation": ["Consolidation"], "Infiltration": ["Infiltration"], "Lung Opacity": ["Lung Opacity"],
        "Nodule/Mass": ["Nodule", "Mass"],
        "Pleural thickening": ["Pleural_Thickening"], "Pulmonary fibrosis": ["Fibrosis"]}


def report(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    p = s >= threshold
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(float(roc_auc_score(y, s)), 3),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def fit(X, y, tr, va, name: str) -> dict:
    """C and threshold from validation only."""
    scaler = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(scaler.transform(X[tr]), y[tr])
        auc = roc_auc_score(y[va], m.predict_proba(scaler.transform(X[va]))[:, 1])
        print(f"{name} C={c} validation AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, c, m)
    _, c, model = best
    s_va = model.predict_proba(scaler.transform(X[va]))[:, 1]
    threshold = float(max(t for t in np.unique(s_va) if (s_va[y[va] == 1] >= t).mean() >= 0.90))
    return {"C": c, "threshold": threshold, "model": model, "scaler": scaler}


def main() -> None:
    d = np.load(DATA / "features_vinbig.npz", allow_pickle=False)
    X, scores, labels, classes = d["X"], d["scores"], d["labels"], d["classes"].tolist()
    col = {c: i for i, c in enumerate(classes)}
    abnormal = labels.any(axis=1).astype(int)
    lung = labels[:, [col["Consolidation"], col["Infiltration"], col["Lung Opacity"]]].any(axis=1).astype(int)
    strata = abnormal + lung  # 0 normal, 1 other problem, 2 lung finding
    all_idx = np.arange(len(X))
    tr, rest = train_test_split(all_idx, test_size=0.30, random_state=42, stratify=strata)
    va, te = train_test_split(rest, test_size=0.50, random_state=42, stratify=strata[rest])
    print(f"train {len(tr)}, validation {len(va)}, test {len(te)}", flush=True)

    original = scores[:, [XRV_LABELS.index(k) for k in ("Pneumonia", "Consolidation", "Infiltration", "Lung Opacity")]].max(axis=1)
    results = {"eval_set": "VinBigData (Vietnam, adults, radiologist labels), held-out 15%: normal, other problems and lung findings"}
    for name, y, file in (("abnormal", abnormal, "abnormal_adult_head.json"), ("lung", lung, "lung_adult_head.json")):
        f = fit(X, y, tr, va, name)
        s = f["model"].predict_proba(f["scaler"].transform(X))[:, 1]
        results[name] = {"chosen_on_validation": {"C": f["C"], "threshold": round(f["threshold"], 4)},
                         "test": report(y[te], s[te], f["threshold"]),
                         "original_model_test": report(y[te], original[te], 0.5)}
        (OUT / file).write_text(json.dumps({
            "mean": f["scaler"].mean_.tolist(), "scale": f["scaler"].scale_.tolist(), "coef": f["model"].coef_[0].tolist(),
            "intercept": float(f["model"].intercept_[0]), "threshold": f["threshold"], "metrics": results[name]}))
    per_finding = {}
    for name, xrv in SAME.items():
        pos = labels[te, col[name]]
        keep = pos | (abnormal[te] == 0)
        s = scores[te][:, [XRV_LABELS.index(k) for k in xrv]].max(axis=1)
        if pos.sum() >= 10:
            per_finding[name] = report(pos[keep].astype(int), s[keep], 0.5)
    results["original_model_per_finding_vs_normal"] = per_finding
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
