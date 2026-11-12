"""Outside test of the type 2 diabetes risk model (Pima, Moizah1) on the Frankfurt hospital diabetes dataset (Kaggle johndasilva/diabetes).
From backend/: python training/verify_pima_frankfurt.py

Rules fixed before scoring (2026-11-12): model, imputer and threshold unchanged, through the app's own adapter, all 8 inputs given
(zeros taken as unknown, as the adapter does). Rows that also appear in the Pima data the model was fitted on (same 8 values) are
removed, and repeated rows are kept once. Scored once. Verified stays if sensitivity and AUC are both >= 0.80.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from workers.tabular.adapters.moizah1_diabetes_prediction import FEATURES, KEYS, Adapter  # noqa: E402

d = pd.read_csv(BACKEND / "data/eval/frankfurt/diabetes.csv")
n0 = len(d)
d = d.drop_duplicates(subset=FEATURES)
a = Adapter()
a.load()
seen = set()
for fold in a.params["folds"]:
    for r in fold["fit_X"]:
        seen.add(tuple(np.round(np.nan_to_num(r, nan=0.0), 3)))
key = lambda r: tuple(np.round([0.0 if (k in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI") and r[k] == 0) else r[k] for k in FEATURES], 3))
mask = np.array([key(r) not in seen for _, r in d.iterrows()])
d = d[mask]
print(f"Frankfurt rows {n0}, unique {int(mask.size)}, also in the Pima training data {int((~mask).sum())}, tested {len(d)}")
p, flag = [], []
for _, r in d.iterrows():
    out = a.predict({k: r[f] for k, f in zip(KEYS, FEATURES, strict=True)} | {"sex": "female"})
    p.append(out.probabilities["Higher risk"])
    flag.append(out.label == "Higher risk")
y, p, flag = d.Outcome.to_numpy(int), np.array(p), np.array(flag)
res = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(flag[y == 1].mean()), 3),
       "specificity": round(float((~flag[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, p)), 3)}
low = min(res["sensitivity"], res["auc"])
res["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
print("OUTSIDE TEST on Frankfurt (once):", res)
(BACKEND / "data/eval/frankfurt/pima_outside.json").write_text(json.dumps(res, indent=1))
