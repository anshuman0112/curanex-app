"""Round 2 Kaggle notebook (CPU, no GPU quota): survey screeners trained on CDC BRFSS, tested once on CDC NHANES 2015-16 - a different
survey with different methods, so a pass is a real outside test. Run from the repo root: python kaggle/make_survey_nhanes_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Curanex: survey screeners round 2 - cancer, skin cancer, vision loss (CPU)

These three failed NHANES 2017-18. **Rules fixed before running (2026-11-11):**
- Inputs: the shared questions as before, plus education (4 levels, both surveys); skin cancer also gets white vs not white
  (a known skin-cancer risk factor that stands in for skin type).
- Train on BRFSS 2022-2025; 15% random hold-out picks logistic vs boosting (AUC) and the threshold (hold-out sensitivity >= 0.82).
- **Outside test, scored once:** NHANES 2015-16 adults, a cycle never used for these models.
- Verified if NHANES sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

CODE = [r'''import io, json, hashlib, os, urllib.request, zipfile
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AGE_MID = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 82}
FEATS = ["age", "male", "bmi", "smoker_current", "smoker_former", "exercise", "general_health", "poor_physical_days", "poor_mental_days",
         "difficulty_walking", "difficulty_concentrating", "diabetes", "heart_disease", "stroke", "asthma", "copd", "kidney_disease", "arthritis"]
yes = lambda s: s.map({1.0: 1.0, 2.0: 0.0})
days = lambda s: s.mask(s == 88, 0).where(lambda v: v <= 30)
RAW = ["SEXVAR", "_AGEG5YR", "_BMI5", "_SMOKER3", "EXERANY2", "GENHLTH", "PHYSHLTH", "MENTHLTH", "DIFFWALK", "DECIDE", "DIABETE4", "_MICHD",
       "CVDSTRK3", "ASTHMA3", "CHCCOPD3", "CHCKDNY2", "HAVARTH4", "HAVARTH5", "CHCSCNC1", "CHCOCNC1", "BLIND", "DEAF", "BPHIGH6", "_EDUCAG", "_RACE"]


def brfss(year):
    url = f"https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"
    path = f"/kaggle/working/b{year}.zip"
    urllib.request.urlretrieve(url, path)
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if n.strip().upper().endswith(".XPT")][0]  # some years add a trailing space
    xpt = f"/kaggle/working/LLCP{year}.XPT"
    with z.open(name) as src, open(xpt, "wb") as dst:
        dst.write(src.read())
    parts = [c.reindex(columns=RAW) for c in pd.read_sas(xpt, format="xport", chunksize=50000)]
    os.remove(xpt); os.remove(path)
    r = pd.concat(parts, ignore_index=True)
    arth = r["HAVARTH4"].fillna(r["HAVARTH5"])
    d = pd.DataFrame({
        "age": r["_AGEG5YR"].map(AGE_MID), "male": r["SEXVAR"].map({1.0: 1.0, 2.0: 0.0}), "bmi": r["_BMI5"] / 100,
        "smoker_current": r["_SMOKER3"].map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0, 4.0: 0.0}), "smoker_former": r["_SMOKER3"].map({1.0: 0.0, 2.0: 0.0, 3.0: 1.0, 4.0: 0.0}),
        "exercise": yes(r["EXERANY2"]), "general_health": r["GENHLTH"].where(r["GENHLTH"] <= 5), "poor_physical_days": days(r["PHYSHLTH"]),
        "poor_mental_days": days(r["MENTHLTH"]), "difficulty_walking": yes(r["DIFFWALK"]), "difficulty_concentrating": yes(r["DECIDE"]),
        "diabetes": r["DIABETE4"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0}), "heart_disease": r["_MICHD"].map({1.0: 1.0, 2.0: 0.0}),
        "stroke": yes(r["CVDSTRK3"]), "asthma": yes(r["ASTHMA3"]), "copd": yes(r["CHCCOPD3"]), "kidney_disease": yes(r["CHCKDNY2"]), "arthritis": yes(arth),
        "skin_cancer": yes(r["CHCSCNC1"]), "cancer": yes(r["CHCOCNC1"]), "vision_loss": yes(r["BLIND"]), "hearing_difficulty": yes(r["DEAF"]),
        "high_blood_pressure": r["BPHIGH6"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0}),
        "education": r["_EDUCAG"].where(r["_EDUCAG"] <= 4), "white": r["_RACE"].where(r["_RACE"] <= 8).map(lambda v: float(v == 1) if v == v else np.nan),
    })
    print(year, len(d), flush=True)
    return d


years = {y: brfss(y) for y in (2022, 2023, 2024, 2025)}''',
r'''def nh(name):
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/{name}_I.xpt"
    return pd.read_sas(io.BytesIO(urllib.request.urlopen(url).read()), format="xport")


d = nh("DEMO")[["SEQN", "RIAGENDR", "RIDAGEYR", "DMDEDUC2", "RIDRETH3"]]
for name in ("BMX", "SMQ", "PAQ", "HSQ", "DLQ", "DIQ", "MCQ", "BPQ", "KIQ_U"):
    d = d.merge(nh(name), on="SEQN", how="left")
col = lambda c: d[c] if c in d else pd.Series(np.nan, index=d.index)
d = d[d.RIDAGEYR >= 18]
age_bin = np.clip(((d.RIDAGEYR - 25) // 5 + 2).where(d.RIDAGEYR >= 25, 1), 1, 13)
smk_now = np.where(col("SMQ020") == 2, 0.0, col("SMQ040").map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0}))
cancer_types = [c for c in ("MCQ230A", "MCQ230B", "MCQ230C", "MCQ230D") if c in d]
types = d[cancer_types]
skin = types.isin([31, 32]).any(axis=1)
other = (types.notna() & ~types.isin([31, 32, 77, 99])).any(axis=1)
had = col("MCQ220")
nhanes = pd.DataFrame({
    "age": age_bin.map(AGE_MID), "male": col("RIAGENDR").map({1.0: 1.0, 2.0: 0.0}), "bmi": col("BMXBMI"),
    "smoker_current": smk_now, "smoker_former": np.where((col("SMQ020") == 1) & (col("SMQ040") == 3), 1.0, np.where(col("SMQ020").isin([1, 2]), 0.0, np.nan)),
    "exercise": np.where((col("PAQ650") == 1) | (col("PAQ665") == 1), 1.0, np.where((col("PAQ650") == 2) & (col("PAQ665") == 2), 0.0, np.nan)),
    "general_health": col("HSD010").where(col("HSD010") <= 5), "poor_physical_days": col("HSQ470").where(col("HSQ470") <= 30),
    "poor_mental_days": col("HSQ480").where(col("HSQ480") <= 30), "difficulty_walking": yes(col("DLQ050")), "difficulty_concentrating": yes(col("DLQ040")),
    "diabetes": col("DIQ010").map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0}), "heart_disease": np.fmax(yes(col("MCQ160C")), yes(col("MCQ160E"))),
    "stroke": yes(col("MCQ160F")), "asthma": yes(col("MCQ010")), "copd": yes(col("MCQ160O")), "kidney_disease": yes(col("KIQ022")),
    "arthritis": yes(col("MCQ160A")),
    "skin_cancer": np.where(had == 2, 0.0, np.where((had == 1) & skin, 1.0, np.nan)),
    "cancer": np.where(had == 2, 0.0, np.where((had == 1) & other, 1.0, np.nan)),
    "education": col("DMDEDUC2").map({1.0: 1.0, 2.0: 1.0, 3.0: 2.0, 4.0: 3.0, 5.0: 4.0}),
    "white": col("RIDRETH3").map(lambda v: float(v == 3) if v == v else np.nan),
    "vision_loss": yes(col("DLQ020")), "hearing_difficulty": yes(col("DLQ010")), "high_blood_pressure": yes(col("BPQ020")),
})
print("NHANES 2015-16 adults:", len(nhanes), {t: int(nhanes[t].sum()) for t in ("skin_cancer", "cancer", "vision_loss", "hearing_difficulty", "high_blood_pressure")})''',
r'''def report(y, p, t):
    f = p >= t
    r = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(f[y == 1].mean()), 3),
         "specificity": round(float((~f[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, p)), 3)}
    low = min(r["sensitivity"], r["auc"]); r["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
    return r


out = {}
for t in ("skin_cancer", "cancer", "vision_loss"):
    feats = [f for f in FEATS if f != t] + ["education"] + (["white"] if t == "skin_cancer" else [])
    ys = (2022, 2023, 2024, 2025)
    a = pd.concat([years[y][years[y][t].notna()] for y in ys], ignore_index=True)
    x, y = a[feats].to_numpy(np.float32), a[t].to_numpy(int)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
    fits = {"logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)).fit(x[tr], y[tr]),
            "boosting": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42).fit(x[tr], y[tr])}
    aucs = {k: roc_auc_score(y[va], m.predict_proba(x[va])[:, 1]) for k, m in fits.items()}
    name = max(aucs, key=aucs.get); model = fits[name]
    pv = model.predict_proba(x[va])[:, 1]; q = np.sort(pv[y[va] == 1]); thr = float(q[int(np.floor(0.18 * len(q)))])
    b = nhanes[nhanes[t].notna()]
    res = report(b[t].to_numpy(int), model.predict_proba(b[feats].to_numpy(np.float32))[:, 1], thr)
    print(f"OUTSIDE TEST on NHANES 2015-16 (once): {t} {name} hold-out AUC {aucs[name]:.3f} -> {res}", flush=True)
    meta = {"target": t, "features": feats, "model": name, "threshold": thr, "metrics": {"validation_auc": round(aucs[name], 3), "test_nhanes_2015": res},
            "trained_on": f"US CDC BRFSS {', '.join(map(str, ys))} survey, {len(tr):,} adults", "test_set": "NHANES 2015-16, a different survey"}
    if name == "logistic":
        imp, sc, clf = (model.steps[i][1] for i in range(3))
        meta |= {"median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    else:
        trees = []
        for pred in model._predictors:
            nd = pred[0].nodes
            trees.append({k: nd[k].tolist() for k in ("feature_idx", "num_threshold", "left", "right", "is_leaf", "value", "missing_go_to_left")})
        meta |= {"trees": trees, "baseline": float(np.ravel(model._baseline_prediction)[0]), "median": np.nanmedian(x[tr], axis=0).tolist()}

        def tree_proba(X):
            raw = np.full(len(X), meta["baseline"])
            for tr_ in trees:
                for r in range(len(X)):
                    n = 0
                    while not tr_["is_leaf"][n]:
                        v = X[r, tr_["feature_idx"][n]]
                        go_left = tr_["missing_go_to_left"][n] if np.isnan(v) else v <= tr_["num_threshold"][n]
                        n = tr_["left"][n] if go_left else tr_["right"][n]
                    raw[r] += tr_["value"][n]
            return 1 / (1 + np.exp(-raw))
        xb = b[feats].to_numpy(np.float32)[:500]
        print("JSON trees max diff vs sklearn on NHANES:", t, float(np.max(np.abs(tree_proba(xb) - model.predict_proba(xb)[:, 1]))), flush=True)
    json.dump(meta, open(f"/kaggle/working/{t}.json", "w"), indent=1)
    out[t] = res
    json.dump(out, open("/kaggle/working/summary.json", "w"), indent=1)
json.dump(out, open("/kaggle/working/summary.json", "w"), indent=1)''']

cells = [{"cell_type": "markdown", "metadata": {}, "source": INTRO},
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "!pip install -q -U onnx skl2onnx onnxruntime"}]
cells += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in CODE]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_survey_round2"
d.mkdir(exist_ok=True)
(d / "survey_round2.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-survey-round2", "title": "Curanex survey round2", "code_file": "survey_round2.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", d)
