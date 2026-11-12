"""Chronic kidney disease from a routine blood and urine report, trained on Indian hospital data, tested once on Bangladesh.
From backend/: python training/train_ckd_india.py

Rules fixed before scoring (2026-11-12):
- Train: UCI Chronic Kidney Disease (Apollo Hospitals, Tamil Nadu, India; Soundarapandian et al. 2015), 400 patients, CKD vs not.
- Outside test, scored once: Risk Factor Prediction of CKD (Enam Medical College, Bangladesh, 2023; UCI), 200 patients, CKD vs not.
  Its values are released as ranges; each range is replaced by its midpoint (open ranges: the bound plus or minus half the
  column's usual range width).
- Inputs (both sets): age, urine specific gravity, urine albumin (dipstick 0-5), haemoglobin, packed cell volume, blood urea,
  serum creatinine, hypertension, diabetes, pedal oedema, poor appetite, anaemia.
- 20% stratified validation from the Indian set; logistic regression (median impute, scaling); threshold = highest with validation
  sensitivity >= 0.85. Verified if test sensitivity and AUC are both >= 0.80; live at >= 0.75.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data/eval/ckd_india_bd"
OUT = BACKEND / "workers/tabular/models/nhanes/ckd_india_report.json"
FEATS = ["age", "sg", "al", "hemo", "pcv", "bu", "sc", "htn", "dm", "pe", "poor_appetite", "ane"]


def india() -> pd.DataFrame:
    d = pd.read_csv(DATA / "uci_ckd_india.csv")
    d = d.apply(lambda s: s.str.strip() if s.dtype == object else s)
    yn = lambda s: s.map({"yes": 1.0, "no": 0.0})
    out = pd.DataFrame({k: pd.to_numeric(d[k], errors="coerce") for k in ["age", "sg", "al", "hemo", "pcv", "bu", "sc"]})
    out["htn"], out["dm"], out["pe"], out["ane"] = yn(d.htn), yn(d.dm), yn(d.pe), yn(d.ane)
    out["poor_appetite"] = d.appet.map({"poor": 1.0, "good": 0.0})
    out["y"] = d.classification.map({"ckd": 1, "notckd": 0})
    return out[out.y.notna()]


def midpoint(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    spans = [tuple(map(float, m.groups())) for v in s for m in [re.fullmatch(r"(-?[\d.]+) - (-?[\d.]+)", v)] if m]
    width = float(np.median([b - a for a, b in spans])) if spans else 0.0

    def one(v: str) -> float:
        if m := re.fullmatch(r"(-?[\d.]+) - (-?[\d.]+)", v):
            return (float(m.group(1)) + float(m.group(2))) / 2
        if m := re.fullmatch(r"<\s*(-?[\d.]+)", v):
            return float(m.group(1)) - width / 2
        if m := re.fullmatch(r"\u2265\s*(-?[\d.]+)", v):
            return float(m.group(1)) + width / 2
        try:
            return float(v)
        except ValueError:
            return np.nan

    return s.map(one)


def bangladesh() -> pd.DataFrame:
    d = pd.read_csv(DATA / "ckd_bangladesh_2023.csv")
    d = d[d["class"].isin(["ckd", "notckd"])]
    out = pd.DataFrame({"age": midpoint(d.age), "sg": midpoint(d.sg), "al": midpoint(d.al).clip(lower=0), "hemo": midpoint(d.hemo),
                        "pcv": midpoint(d.pcv), "bu": midpoint(d.bu), "sc": midpoint(d.sc)})
    for k, src in (("htn", "htn"), ("dm", "dm"), ("pe", "pe"), ("ane", "ane"), ("poor_appetite", "appet")):
        out[k] = pd.to_numeric(d[src], errors="coerce")
    out["y"] = (d["class"] == "ckd").astype(int)
    return out


def report(y, p, t):
    f = p >= t
    r = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(f[y == 1].mean()), 3),
         "specificity": round(float((~f[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, p)), 3)}
    low = min(r["sensitivity"], r["auc"])
    r["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
    return r


def main() -> None:
    a, b = india(), bangladesh()
    print("India:", len(a), int(a.y.sum()), "| Bangladesh:", len(b), int(b.y.sum()), "| missing in test:", {k: round(float(b[k].isna().mean()), 2) for k in FEATS})
    x, y = a[FEATS].to_numpy(float), a.y.to_numpy(int)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)).fit(x[tr], y[tr])
    pv = model.predict_proba(x[va])[:, 1]
    thr = float(max(v for v in np.unique(np.round(pv, 5)) if (pv[y[va] == 1] >= v).mean() >= 0.85))
    res = report(b.y.to_numpy(int), model.predict_proba(b[FEATS].to_numpy(float))[:, 1], thr)
    print("validation AUC", round(roc_auc_score(y[va], pv), 3), "| OUTSIDE TEST on Bangladesh (once):", res)
    imp, sc, clf = (model.steps[i][1] for i in range(3))
    meta = {"target": "ckd_india_report", "features": FEATS, "model": "logistic", "threshold": thr, "min_age": 12, "passed": res["status"] == "verified",
            "metrics": {"validation_auc": round(float(roc_auc_score(y[va], pv)), 3), "test": res},
            "trained_on": f"UCI Chronic Kidney Disease (Apollo Hospitals, Tamil Nadu, India), {len(tr):,} people",
            "test_set": "Enam Medical College, Bangladesh (2023), different hospital and country",
            "impute": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]), "median": imp.statistics_.tolist()}
    OUT.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
