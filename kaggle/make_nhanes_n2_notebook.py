"""NHANES round K (2026-11-12): peripheral artery disease, diabetic foot nerve damage, depression. Run: python kaggle/make_nhanes_k_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTRO = """# Curanex: NHANES round N - low fitness and balance problems (CPU)

**Rules fixed before running (2026-11-12):** truth measured; the measuring test is never an input.
- Low cardiorespiratory fitness: NHANES treadmill fitness level "low" (CVDFITLV = 1, age- and sex-specific), ages 12-49.
  Train 1999-2000 + 2001-02; **outside test, scored once:** 2003-04.
- Balance problem: failed the hardest Romberg condition (eyes closed on foam), age 40+. Train 2001-02; **outside test, scored once:** 2003-04.
- Inputs (questions only): age, sex, BMI, smoking, told diabetes, heart disease, general health, vigorous and moderate activity
  (+ dizziness question for balance). 20% stratified validation; logistic unless boosting wins by 0.02; threshold = highest with
  validation sensitivity >= 0.85. Verified if sensitivity and AUC are both >= 0.80."""

CODE = r'''import io, json, re, urllib.request
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
YEAR = {"A": 1999, "B": 2001, "C": 2003, "G": 2011, "H": 2013, "J": 2017, "L": 2021}
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


def q(c, extra):
    d, col, f = base(c, ["BMX", "SMQ", "DIQ", "BPQ", "MCQ", "HSQ", "PAQ", "BAQ"] + extra)
    out = pd.DataFrame({"age": f.age, "male": f.male, "bmi": f.bmi, "smoker_current": f.smoker_current, "diabetes": f.diabetes,
                        "heart_disease": f.heart_disease, "general_health": col("HSD010").where(col("HSD010") <= 5),
                        "vigorous_activity": yes(col("PAD200")), "moderate_activity": yes(col("PAD320"))})
    return d, col, out


def fitness(c):
    d, col, out = q(c, ["CVX"])
    lv = col("CVDFITLV")  # 1 low, 2 moderate, 3 high (age- and sex-specific)
    out["y"] = np.where(lv.isin([1, 2, 3]), (lv == 1).astype(float), np.nan)
    return out[out.y.notna()]


def balance(c):
    d, col, out = q(c, ["BAX"])
    print("balance columns", c, [k for k in d.columns if k.startswith("BAX")][:12])
    r = col("BAXPFC41")  # condition 4 (eyes closed, foam), pass/fail
    print("condition 4 codes", c, r.value_counts().to_dict())
    out["dizziness"] = yes(col("BAQ321C")) if "BAQ321C" in d else np.nan
    out["y"] = np.where(r.isin([1, 2]) & (d.RIDAGEYR >= 40), (r == 2).astype(float), np.nan)
    return out[out.y.notna()]


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


run("balance_problem_questions", balance("B"), balance("C"), "2001-02", "2003-04", 40)'''

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": INTRO},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_nhanes_n2"
d.mkdir(exist_ok=True)
(d / "nhanes_n.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-nhanes-n2", "title": "Curanex nhanes n2", "code_file": "nhanes_n.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", d)
