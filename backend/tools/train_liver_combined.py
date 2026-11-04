"""Liver check on German + Indian data together. Tabular image, no network:

  docker run --rm --network none -v <backend>/data/eval:/data:ro -v <models>/yahyazuher_ai_liver_disease_diagnosis_system:/out \
      -v <backend>/tools:/tools:ro backend-worker-tabular python /tools/train_liver_combined.py

UCI HCV + ILPD on the tests both share (GGT dropped, as ILPD lacks it). 30% of each held out (seed 42), duplicates removed
from ILPD first so a row can't sit on both sides. Choice of model, C and threshold (highest with sensitivity >= 0.90) come
from 5-fold out-of-fold predictions on training rows; each held-out part is scored once, ILPD (India) being the one that matters.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("/out")
LOG = ["ALP", "ALT", "AST", "BIL"]
COLS = ["Age", "male", "ALB", "PROT", *LOG]


def report(y, s, t):
    p = s >= t
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3), "auc": round(float(roc_auc_score(y, s)), 3)}


def main() -> None:
    h = pd.read_csv("/data/hcv/hcvdat0.csv", index_col=0)
    h = h[~h["Category"].str.startswith("0s")]
    hcv = pd.DataFrame({"Age": h["Age"], "male": (h["Sex"] == "m") * 1.0, "ALB": h["ALB"], "PROT": h["PROT"], "ALP": h["ALP"], "ALT": h["ALT"],
                        "AST": h["AST"], "BIL": h["BIL"], "y": (~h["Category"].str.startswith("0")).astype(int)})
    i = pd.read_csv("/data/ilpd/ilpd.csv", header=None).drop_duplicates()
    ilpd = pd.DataFrame({"Age": i[0], "male": (i[1] == "Male") * 1.0, "ALB": i[8] * 10, "PROT": i[7] * 10, "ALP": i[4], "ALT": i[5], "AST": i[6],
                         "BIL": i[2] * 17.1, "y": (i[10] == 1).astype(int)})
    frames = []
    for name, d in (("hcv", hcv), ("ilpd", ilpd)):
        tr, te = train_test_split(np.arange(len(d)), test_size=0.30, random_state=42, stratify=d["y"])
        d = d.reset_index(drop=True).assign(source=name, split="train")
        d.loc[te, "split"] = "test"
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    train = d["split"] == "train"
    medians = d.loc[train, COLS].median()
    Z = d[COLS].fillna(medians).astype(float)
    Z[LOG] = np.log1p(Z[LOG])
    y = d["y"].values

    candidates = {f"logistic C={c}": make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000)) for c in (0.1, 1.0)}
    candidates["gradient boosting"] = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)
    best = None
    for name, model in candidates.items():
        oof = cross_val_predict(model, Z[train], y[train], cv=5, method="predict_proba")[:, 1]
        ilpd_rows = d.loc[train, "source"].values == "ilpd"
        auc = roc_auc_score(y[train][ilpd_rows], oof[ilpd_rows])
        print(f"{name}: training-fold AUC on Indian rows {auc:.3f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, name, model, oof, ilpd_rows)
    _, name, model, oof, ilpd_rows = best
    threshold = float(max(t for t in np.unique(oof[ilpd_rows]) if (oof[ilpd_rows][y[train][ilpd_rows] == 1] >= t).mean() >= 0.90))
    model.fit(Z[train], y[train])
    s = model.predict_proba(Z)[:, 1]
    out = {"model": name, "threshold": round(threshold, 4)}
    for src in ("ilpd", "hcv"):
        m = (d["source"] == src).values & ~train.values
        out[src] = report(y[m], s[m], threshold)
    print(json.dumps(out, indent=1))
    scaler, clf = model[0], model[1]
    (OUT / "model_meta.json").write_text(json.dumps({
        "columns": COLS, "log": LOG, "medians": medians.tolist(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "threshold": threshold, "metrics": out}, indent=1))


if __name__ == "__main__":
    main()
