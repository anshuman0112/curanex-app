"""Kaggle CPU notebook: hearing loss and low bone density retrained on more NHANES cycles, each tested once on a cycle never used.
Run from the repo root: python kaggle/make_nhanes_round2_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Curanex: NHANES round 2 - hearing loss and low bone density (CPU)

**Rules fixed before running (2026-11-11):**
- **Hearing loss** (better-ear average of 0.5/1/2/4 kHz > 25 dB): inputs age, sex, self-rated hearing, hears a whisper, hears a normal
  voice, ringing in the ears. Train on NHANES 2011-12 + 2015-16 (ages 20-69, the old model's train and test cycles).
  **Outside test, scored once:** NHANES 2017-18 adults (ages 70-80 had audiometry that cycle), never used before.
- **Low bone density** (femoral neck T-score <= -1, age 50+): inputs age, sex, weight, height, smoking (now/ever), diabetes, plus
  broken hip / wrist / spine, ever took prednisone or cortisone daily, mother or father broke a hip. Train on NHANES 2009-10, 2013-14,
  2017-18. **Outside test, scored once:** NHANES 2005-06 + 2007-08, never used before.
- Both: 20% stratified validation; logistic regression unless boosting wins by 0.02 validation AUC; threshold = highest with validation
  sensitivity >= 0.85. Verified if test sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

CODE = [r'''import io, json, urllib.request
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

YEAR = {"D": 2005, "E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017}
yes = lambda s: s.map({1.0: 1.0, 2.0: 0.0})


def nh(name, c):
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{YEAR[c]}/DataFiles/{name}_{c}.xpt"
    try:
        return pd.read_sas(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()), format="xport")
    except Exception as e:
        print("missing", name, c, e); return pd.DataFrame({"SEQN": []})


def cycle(c, files):
    d = nh("DEMO", c)[["SEQN", "RIAGENDR", "RIDAGEYR"]]
    for f in files:
        d = d.merge(nh(f, c), on="SEQN", how="left")
    col = lambda k: d[k] if k in d else pd.Series(np.nan, index=d.index)
    return d, col


def hearing(c):
    d, col = cycle(c, ["AUQ", "AUX"])
    def pta(s):
        t = d[[f"AUXU500{s}", f"AUXU1K1{s}", f"AUXU2K{s}", f"AUXU4K{s}"]]
        return t.where(t <= 120).mean(axis=1, skipna=False)
    better = np.fmin(pta("R"), pta("L"))
    f = pd.DataFrame({"age": d.RIDAGEYR, "male": d.RIAGENDR.map({1.0: 1.0, 2.0: 0.0}), "hearing_rating": col("AUQ054").where(lambda v: v <= 6),
                      "hears_whisper": yes(col("AUQ060")), "hears_normal_voice": yes(col("AUQ070")), "ringing_ears": yes(col("AUQ191"))})
    f["y"] = np.where(better.notna(), (better > 25).astype(float), np.nan)
    return f[f.y.notna() & (f.age >= 20)]


def bone(c):
    d, col = cycle(c, ["BMX", "SMQ", "DIQ", "OSQ", "DXXFEM"])
    t1 = (col("DXXNKBMD") - 0.858) / 0.120
    f = pd.DataFrame({"age": d.RIDAGEYR, "male": d.RIAGENDR.map({1.0: 1.0, 2.0: 0.0}), "weight_kg": col("BMXWT"), "height_cm": col("BMXHT"),
                      "smoker_current": np.where(col("SMQ020") == 2, 0.0, col("SMQ040").map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0})),
                      "smoker_ever": yes(col("SMQ020")), "diabetes": col("DIQ010").map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0}),
                      "broke_hip": yes(col("OSQ010A")), "broke_wrist": yes(col("OSQ010B")), "broke_spine": yes(col("OSQ010C")),
                      "steroids": yes(col("OSQ130")), "parent_hip_fracture": np.fmax(yes(col("OSQ170")), yes(col("OSQ200")))})
    f["y"] = np.where(col("DXXNKBMD").notna(), (t1 <= -1.0).astype(float), np.nan)
    return f[f.y.notna() & (f.age >= 50)]''',
r'''def report(y, p, t):
    f = p >= t
    r = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(f[y == 1].mean()), 3),
         "specificity": round(float((~f[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, p)), 3)}
    low = min(r["sensitivity"], r["auc"]); r["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
    return r


def run(target, fn, train_cycles, test_cycles, train_label, test_label):
    a = pd.concat([fn(c) for c in train_cycles], ignore_index=True)
    b = pd.concat([fn(c) for c in test_cycles], ignore_index=True)
    feats = [k for k in a.columns if k != "y"]
    print(target, "train", len(a), int(a.y.sum()), "| test", len(b), int(b.y.sum()), "| missing in test:", {k: round(float(b[k].isna().mean()), 2) for k in feats}, flush=True)
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
    print(f"OUTSIDE TEST on {test_label} (once): {target} {name} val AUC {aucs[name]:.3f} -> {res}", flush=True)
    meta = {"target": target, "features": feats, "model": name, "threshold": thr, "min_age": 20 if target == "hearing_loss" else 50,
            "metrics": {"validation_auc": round(aucs[name], 3), "test_fresh2": res | {"cycle": test_label}},
            "trained_on": f"US CDC NHANES {train_label}, {len(tr):,} people", "test_set": f"NHANES {test_label}, never used before",
            "median": np.nanmedian(x[tr], axis=0).tolist(), "train_min": np.nanmin(x[tr], axis=0).tolist(), "train_max": np.nanmax(x[tr], axis=0).tolist()}
    if name == "logistic":
        imp, sc, clf = (model.steps[i][1] for i in range(3))
        meta |= {"impute": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    else:
        meta |= {"trees": [{k: p[0].nodes[k].tolist() for k in ("feature_idx", "num_threshold", "left", "right", "is_leaf", "value", "missing_go_to_left")} for p in model._predictors],
                 "baseline": float(np.ravel(model._baseline_prediction)[0])}
    json.dump(meta, open(f"/kaggle/working/{target}.json", "w"))
    return res


out = {"hearing_loss": run("hearing_loss", hearing, ["G", "I"], ["J"], "2011-12 + 2015-16", "2017-18 (ages 70-80)"),
       "low_bone_density": run("low_bone_density", bone, ["F", "H", "J"], ["D", "E"], "2009-10, 2013-14, 2017-18", "2005-06 + 2007-08")}
json.dump(out, open("/kaggle/working/summary.json", "w"), indent=1)''']

cells = [{"cell_type": "markdown", "metadata": {}, "source": INTRO}]
cells += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in CODE]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_nhanes_round2"
d.mkdir(exist_ok=True)
(d / "nhanes_round2.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-nhanes-round2", "title": "Curanex nhanes round2", "code_file": "nhanes_round2.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", d)
