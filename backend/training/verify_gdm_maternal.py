"""Outside test of the gestational diabetes check on the Maternal Health and High-Risk Pregnancy dataset (Kaggle,
vmohammedraiyyan, CC BY-NC-SA 4.0). From backend/: python training/verify_gdm_maternal.py

Rules fixed before scoring (2026-11-11): the live model and its "few inputs" threshold are used unchanged, through the app's own
adapter, with the three inputs the app asks for: age, BMI and blood sugar (BS, mmol/L, taken as fasting glucose). Pregnancies with
pre-existing diabetes are left out (that is not gestational diabetes); rows missing age, BMI or sugar are left out. Scored once.
Verified if sensitivity and AUC are both >= 0.80; live at >= 0.75.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from workers.tf.adapters.srishti_chordia_gestational_diabetes_prediction_using_dl import Adapter  # noqa: E402

d = pd.read_csv(BACKEND / "data/eval/maternal_risk/maternal_health_risk.csv")
d = d[(d["Preexisting Diabetes"] == 0) & d[["Age", "BMI", "BS"]].notna().all(axis=1) & d["Gestational Diabetes"].notna()]
a = Adapter()
a.load()
p, flag = [], []
for r in d.itertuples():
    out = a.predict({"age": int(r.Age), "bmi": float(np.clip(r.BMI, 12, 60)), "fasting_glucose": float(np.clip(r.BS * 18, 40, 400))})
    p.append(out.probabilities["Higher risk"])
    flag.append(out.label == "Higher risk")
y, p, flag = d["Gestational Diabetes"].to_numpy(int), np.array(p), np.array(flag)
res = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(flag[y == 1].mean()), 3),
       "specificity": round(float((~flag[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, p)), 3)}
low = min(res["sensitivity"], res["auc"])
res["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
print("OUTSIDE TEST on maternal health dataset (once):", res)
(BACKEND / "data/eval/maternal_risk/gdm_outside.json").write_text(json.dumps(res, indent=1))
