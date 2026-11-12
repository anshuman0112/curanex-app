"""NHANES round K (2026-11-12): peripheral artery disease, diabetic foot nerve damage, depression. Run: python kaggle/make_nhanes_k_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTRO = """# Curanex: NHANES round K - peripheral artery disease, foot nerve damage, depression (CPU)

**Rules fixed before running (2026-11-12):** truth measured; the measuring test is never an input.
- Peripheral artery disease: ankle-brachial index < 0.90 in either leg, age 40+. Foot nerve damage: NHANES derived abnormal monofilament result
  (LEALPN / LEARPN = 2) in either foot, both feet tested, age 40+. Both from questions only (age, sex, BMI, smoking now/ever, told diabetes, told high BP,
  told high cholesterol, heart disease, stroke). Train NHANES 1999-2000 + 2001-02; **outside test, scored once:** 2003-04.
- Current depression: PHQ-9 total >= 10, age 18+, from questions that are not PHQ items (age, sex, BMI, general health, told trouble
  sleeping, smoking, told diabetes, told high BP). Train 2017-18; **outside test, scored once:** 2021-23.
- 20% stratified validation; logistic unless boosting wins by 0.02; threshold = highest with validation sensitivity >= 0.85.
  Verified if sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

CODE = r'''import io, json, urllib.request
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
YEAR = {"A": 1999, "B": 2001, "C": 2003, "J": 2017, "L": 2021}
yes = lambda s: s.map({1.0: 1.0, 2.0: 0.0})


def nh(name, c):
    file = {"LEXAB": "LEXABPI"}.get(name, name) if c == "A" else f"{name}_{c}"
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{YEAR[c]}/DataFiles/{file}.xpt"
    try:
        return pd.read_sas(io.BytesIO(urllib.request.urlopen(url, timeout=180).read()), format="xport")
    except Exception as e:
        print("missing", file, e)
        return pd.DataFrame({"SEQN": []})


def base(c, files):
    d = nh("DEMO", c)[["SEQN", "RIAGENDR", "RIDAGEYR"]]
    for f in files:
        x = nh(f, c)
        if "SEQN" in x:
            d = d.merge(x[["SEQN"] + [k for k in x.columns if k != "SEQN" and k not in d.columns]], on="SEQN", how="left")
    col = lambda k: d[k] if k in d else pd.Series(np.nan, index=d.index)
    f = pd.DataFrame({"age": d.RIDAGEYR, "male": d.RIAGENDR.map({1.0: 1.0, 2.0: 0.0}), "bmi": col("BMXBMI"),
                      "smoker_current": np.where(col("SMQ020") == 2, 0.0, col("SMQ040").map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0})),
                      "smoker_ever": yes(col("SMQ020")), "diabetes": col("DIQ010").map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0}),
                      "high_bp": yes(col("BPQ020")), "high_chol": yes(col("BPQ080")),
                      "heart_disease": np.fmax(yes(col("MCQ160C")), yes(col("MCQ160E"))), "stroke": yes(col("MCQ160F"))})
    return d, col, f


def pad(c):
    d, col, f = base(c, ["BMX", "SMQ", "DIQ", "BPQ", "MCQ", "LEXAB"])
    abi = np.fmin(col("LEXLABPI"), col("LEXRABPI"))
    f["y"] = np.where(abi.notna() & (d.RIDAGEYR >= 40), (abi < 0.90).astype(float), np.nan)
    return f[f.y.notna()]


def neuro(c):
    d, col, f = base(c, ["BMX", "SMQ", "DIQ", "BPQ", "MCQ", "LEXPN"])
    sites = [k for k in d.columns if k.startswith(("LEALPN", "LEARPN"))]
    print("monofilament columns", c, sites[:4], len(sites))
    if not sites:
        return f.iloc[0:0].assign(y=[])
    vals = d[sites].where(d[sites].isin([1, 2]))  # 1 = felt, 2 = not felt
    f["y"] = np.where((vals.notna().sum(axis=1) == len(sites)) & (d.RIDAGEYR >= 40), (vals == 2).any(axis=1).astype(float), np.nan)
    return f[f.y.notna()]


def depression(c):
    d, col, f = base(c, ["BMX", "SMQ", "DIQ", "BPQ", "MCQ", "DPQ", "HSQ", "HUQ", "SLQ"])
    items = d[[f"DPQ0{i}0" for i in range(1, 10) if f"DPQ0{i}0" in d]]
    items = items.where(items <= 3)
    gh = col("HUQ010") if "HUQ010" in d else col("HSD010")
    g = pd.DataFrame({"age": f.age, "male": f.male, "bmi": f.bmi, "general_health": gh.where(gh <= 5), "trouble_sleeping": yes(col("SLQ050")),
                      "smoker_current": f.smoker_current, "diabetes": f.diabetes, "high_bp": f.high_bp})
    full = items.notna().sum(axis=1) == items.shape[1]
    g["y"] = np.where(full & (d.RIDAGEYR >= 18), (items.sum(axis=1) >= 10).astype(float), np.nan)
    return g[g.y.notna()]


def report(y, p, t):
    fl = p >= t
    r = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(fl[y == 1].mean()), 3),
         "specificity": round(float((~fl[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, p)), 3)}
    low = min(r["sensitivity"], r["auc"])
    r["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
    return r


def run(target, a, b, train_label, test_label, min_age):
    feats = [k for k in a.columns if k != "y"]
    print(target, "train", len(a), int(a.y.sum()), "| test", len(b), int(b.y.sum()), flush=True)
    if len(b) < 300 or b.y.sum() < 20 or len(a) < 500:
        print("skipped (too little data):", target)
        return None
    x, y = a[feats].to_numpy(np.float32), a.y.to_numpy(int)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    fits = {"logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)).fit(x[tr], y[tr]),
            "boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=42).fit(x[tr], y[tr])}
    aucs = {k: roc_auc_score(y[va], m.predict_proba(x[va])[:, 1]) for k, m in fits.items()}
    name = "boosting" if aucs["boosting"] > aucs["logistic"] + 0.02 else "logistic"
    model = fits[name]
    pv = model.predict_proba(x[va])[:, 1]
    thr = float(max(v for v in np.unique(np.round(pv, 5)) if (pv[y[va] == 1] >= v).mean() >= 0.85))
    res = report(b.y.to_numpy(int), model.predict_proba(b[feats].to_numpy(np.float32))[:, 1], thr)
    print(f"OUTSIDE TEST on NHANES {test_label} (once): {target} {name} val AUC {aucs[name]:.3f} -> {res}", flush=True)
    meta = {"target": target, "features": feats, "model": name, "threshold": thr, "min_age": min_age,
            "metrics": {"validation_auc": round(aucs[name], 3), "test": res}, "trained_on": f"US CDC NHANES {train_label}, {len(tr):,} people",
            "test_set": f"NHANES {test_label}, different people", "median": np.nanmedian(x[tr], axis=0).tolist(),
            "train_min": np.nanmin(x[tr], axis=0).tolist(), "train_max": np.nanmax(x[tr], axis=0).tolist()}
    if name == "logistic":
        imp, sc, clf = (model.steps[i][1] for i in range(3))
        meta |= {"impute": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    else:
        meta |= {"trees": [{k: p[0].nodes[k].tolist() for k in ("feature_idx", "num_threshold", "left", "right", "is_leaf", "value", "missing_go_to_left")} for p in model._predictors],
                 "baseline": float(np.ravel(model._baseline_prediction)[0])}
    json.dump(meta, open(f"/kaggle/working/{target}.json", "w"))
    return res


run("foot_neuropathy_questions", pd.concat([neuro("A"), neuro("B")], ignore_index=True), neuro("C"), "1999-2002", "2003-04", 40)
'''

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": INTRO},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_nhanes_k2"
d.mkdir(exist_ok=True)
(d / "nhanes_k.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-nhanes-k2", "title": "Curanex nhanes k2", "code_file": "nhanes_k.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", d)
