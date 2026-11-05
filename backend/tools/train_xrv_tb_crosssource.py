"""Third TB pass: train on TBX11K (healthy, sick without TB, TB), test on NLM and Belarus + RSNA, which it never saw.
Run from backend/: python tools/train_xrv_tb_crosssource.py

The NLM-trained layer scored AUC 0.54 on TBX11K, so it had learned its source. Rules fixed first: C by 5-fold AUC on TBX11K;
threshold from out-of-fold scores, highest with sensitivity >= 0.85; scored once on the outside sources; ships only if
both outside tests reach sensitivity >= 0.80 and AUC >= 0.80. Nothing is written unless it passes.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "eval"
OUT = BACKEND / "workers" / "torch" / "models" / "mlmed_torchxrayvision" / "tb_adult_head.json"


def report(y, s, t):
    tn, fp, fn, tp = confusion_matrix(y, s >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, s)), 3)}


def main() -> None:
    d = np.load(DATA / "features_tbx.npz", allow_pickle=False)
    X, y = d["X"], (d["kind"] == "tb").astype(int)
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        oof = np.zeros(len(y))
        for a, b in folds.split(X, y):
            oof[b] = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000)).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        auc = roc_auc_score(y, oof)
        print(f"C={c} 5-fold AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, c, oof)
    _, c, oof = best
    t = float(max(v for v in np.unique(oof) if (oof[y == 1] >= v).mean() >= 0.85))
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(scaler.transform(X), y)
    o = np.load(DATA / "features_tb.npz", allow_pickle=False)
    s = model.predict_proba(scaler.transform(o["X"]))[:, 1]
    nlm, bel, rsna = o["source"] == "nlm", o["source"] == "belarus", o["source"] == "rsna"
    results = {"C": c, "threshold": t, "five_fold": report(y, oof, t), "outside_nlm": report(o["y"][nlm], s[nlm], t),
               "outside_belarus_vs_rsna": report(o["y"][bel | rsna], s[bel | rsna], t)}
    print(json.dumps(results, indent=0), flush=True)
    a, b = results["outside_nlm"], results["outside_belarus_vs_rsna"]
    passed = min(a["sensitivity"], b["sensitivity"]) >= 0.80 and min(a["auc"], b["auc"]) >= 0.80
    print("PASSED" if passed else "FAILED", flush=True)
    if passed:
        OUT.write_text(json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist(),
                                   "intercept": float(model.intercept_[0]), "threshold": t, "metrics": results}))
    elif OUT.exists():
        OUT.unlink()  # an earlier layer failed on new data and must not be loaded by mistake


if __name__ == "__main__":
    main()
