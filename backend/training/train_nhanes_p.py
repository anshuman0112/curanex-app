"""NHANES round P (2026-11-14): replacements for the three Unverified checks (heart risk, underactive thyroid, chest X-ray).
Run from backend/: python training/train_nhanes_p.py

Rules fixed before any outcome data was downloaded (2026-11-14):
- Truth is measured; the measuring test is never an input. Inputs are questions a patient can answer (and height/weight/waist).
- Train on the listed training cycles; 20% stratified validation; logistic regression unless histogram gradient boosting wins by
  more than 0.02 validation AUC; threshold = highest with validation sensitivity >= 0.85.
- Outside test on a later NHANES cycle (different people, different years), scored ONCE.
  Verified only if test sensitivity >= 0.80 AND AUC >= 0.80 (at least 300 people and 20 cases, else skipped).
- Deployment order if they pass (up to three replacements): 1, 2, 3. A target that fails is not retried in this round.

Targets:
1. low_muscle_mass: DXA appendicular lean mass / BMI below 0.789 (men) or 0.512 (women) (FNIH sarcopenia project, Studenski 2014),
   ages 20-59 (NHANES whole-body DXA age range). Inputs: age, sex, height, weight (BMI), waist, exercise, general health,
   told diabetes, smoking. Train 2011-12 + 2013-14; test 2017-18.
2. heart_strain_ntprobnp: NT-proBNP >= 125 pg/mL (ESC non-acute heart failure rule-out cut-off), age 20+. Inputs: age, sex, BMI,
   told high BP, told diabetes, heart attack or coronary disease, heart failure, stroke, smoking. Train 1999-2000 + 2001-02;
   test 2003-04 (cycle from SEQN). If the surplus-serum file does not cover 2003-04, the target is skipped before scoring.
3. low_testosterone: total testosterone < 300 ng/dL (AUA 2018), men 20+. Inputs: age, height, weight (BMI), waist, told diabetes,
   told high BP, smoking, general health, exercise. Train 2011-12 + 2013-14; test 2015-16.
"""

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DIR = BACKEND / "data" / "eval" / "nhanes_p"
OUT = DIR / "out"
YEAR = {"A": 1999, "B": 2001, "C": 2003, "G": 2011, "H": 2013, "I": 2015, "J": 2017}
CYCLE_YEARS = {"A": "1999-2000", "B": "2001-02", "C": "2003-04", "G": "2011-12", "H": "2013-14", "I": "2015-16", "J": "2017-18"}
yes = lambda s: s.map({1.0: 1.0, 2.0: 0.0})  # noqa: E731


def nh(name: str, c: str) -> pd.DataFrame:
    file = name if c == "A" and name not in ("SSBNP",) else f"{name}_{c}"
    path = DIR / f"{file}.xpt"
    if not path.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = urllib.request.urlopen(f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{YEAR[c]}/DataFiles/{file}.xpt", timeout=300).read()
        except Exception as e:  # noqa: BLE001
            print("missing", file, e)
            return pd.DataFrame({"SEQN": []})
        path.write_bytes(data)
    return pd.read_sas(path, format="xport", encoding="latin-1")


def merged(c: str, files: list[str]) -> pd.DataFrame:
    d = nh("DEMO", c)[["SEQN", "RIAGENDR", "RIDAGEYR"]]
    for f in files:
        x = nh(f, c)
        if "SEQN" in x:
            d = d.merge(x[["SEQN"] + [k for k in x.columns if k != "SEQN" and k not in d.columns]], on="SEQN", how="left")
    return d


def col(d: pd.DataFrame, k: str) -> pd.Series:
    return d[k] if k in d else pd.Series(np.nan, index=d.index)


def common(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "age": d.RIDAGEYR, "male": d.RIAGENDR.map({1.0: 1.0, 2.0: 0.0}), "height_cm": col(d, "BMXHT"), "weight_kg": col(d, "BMXWT"),
        "bmi": col(d, "BMXBMI"), "waist_cm": col(d, "BMXWAIST"),
        "smoker_current": np.where(col(d, "SMQ020") == 2, 0.0, col(d, "SMQ040").map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0})),
        "diabetes": col(d, "DIQ010").map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0}), "high_bp": yes(col(d, "BPQ020")),
        "general_health": col(d, "HSD010").where(col(d, "HSD010") <= 5),
        "exercise": np.where((col(d, "PAQ650") == 1) | (col(d, "PAQ665") == 1), 1.0,
                             np.where((col(d, "PAQ650") == 2) & (col(d, "PAQ665") == 2), 0.0, np.nan)),
        "heart_attack_or_chd": np.fmax(yes(col(d, "MCQ160C")), yes(col(d, "MCQ160E"))),
        "heart_failure": yes(col(d, "MCQ160B")), "stroke": yes(col(d, "MCQ160F")),
    })


def muscle(c: str) -> pd.DataFrame:
    d = merged(c, ["BMX", "SMQ", "DIQ", "HSQ", "PAQ", "DXX"])
    f = common(d)
    alm = (col(d, "DXDLALE") + col(d, "DXDRALE") + col(d, "DXDLLLE") + col(d, "DXDRLLE")) / 1000
    ratio = alm / f.bmi
    cut = np.where(d.RIAGENDR == 1, 0.789, 0.512)
    f["y"] = np.where(ratio.notna() & (d.RIDAGEYR >= 20) & (d.RIDAGEYR <= 59), (ratio < cut).astype(float), np.nan)
    return f[["age", "male", "height_cm", "weight_kg", "bmi", "waist_cm", "exercise", "general_health", "diabetes", "smoker_current", "y"]]


def ntprobnp() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    b = nh("SSBNP", "A")
    print("SSBNP columns", list(b.columns), "SEQN range", b.SEQN.min(), b.SEQN.max(), flush=True)
    val = "SSBNP"  # NT-proBNP, pg/mL (the file also holds weights and limit-of-detection flags)
    parts = []
    for c in ("A", "B", "C"):
        d = merged(c, ["BMX", "SMQ", "DIQ", "BPQ", "MCQ"]).merge(b[["SEQN", val]], on="SEQN", how="inner")
        f = common(d)
        f["y"] = np.where(d[val].notna() & (d.RIDAGEYR >= 20), (d[val] >= 125).astype(float), np.nan)
        parts.append(f.assign(cycle=c)[["age", "male", "bmi", "high_bp", "diabetes", "heart_attack_or_chd", "heart_failure", "stroke", "smoker_current", "y", "cycle"]])
    a = pd.concat(parts[:2])
    t = parts[2]
    if t.y.notna().sum() < 300:
        print("heart_strain_ntprobnp: the surplus-serum file does not cover 2003-04 - skipped before scoring", flush=True)
        return None
    return a.drop(columns="cycle"), t.drop(columns="cycle")


def testosterone(c: str) -> pd.DataFrame:
    d = merged(c, ["BMX", "SMQ", "DIQ", "BPQ", "HSQ", "PAQ", "TST"])
    f = common(d)
    f["y"] = np.where(col(d, "LBXTST").notna() & (d.RIAGENDR == 1) & (d.RIDAGEYR >= 20), (col(d, "LBXTST") < 300).astype(float), np.nan)
    return f[["age", "height_cm", "weight_kg", "bmi", "waist_cm", "diabetes", "high_bp", "smoker_current", "general_health", "exercise", "y"]]


def report(y, p, t):
    fl = p >= t
    return {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(fl[y == 1].mean()), 3),
            "specificity": round(float((~fl[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, p)), 3)}


def run(target, a, b, train_label, test_label, min_age):
    a, b = a[a.y.notna()], b[b.y.notna()]
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
    res["passed"] = bool(res["sensitivity"] >= 0.80 and res["auc"] >= 0.80)
    print(f"OUTSIDE TEST on NHANES {test_label} (once): {target} {name} val AUC lr {aucs['logistic']:.3f} gb {aucs['boosting']:.3f} -> {res}", flush=True)
    meta = {"target": target, "features": feats, "model": name, "threshold": thr, "passed": res["passed"], "min_age": min_age,
            "metrics": {"validation_auc": round(aucs[name], 3), "test": res}, "trained_on": f"US CDC NHANES {train_label}, {len(tr):,} people",
            "test_set": f"NHANES {test_label}, different people", "median": np.nanmedian(x[tr], axis=0).tolist(),
            "train_min": np.nanmin(x[tr], axis=0).tolist(), "train_max": np.nanmax(x[tr], axis=0).tolist()}
    if name == "logistic":
        imp, sc, clf = (model.steps[i][1] for i in range(3))
        meta |= {"impute": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    else:
        meta |= {"trees": [{k: p[0].nodes[k].tolist() for k in ("feature_idx", "num_threshold", "left", "right", "is_leaf", "value", "missing_go_to_left")} for p in model._predictors],
                 "baseline": float(np.ravel(model._baseline_prediction)[0])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{target}.json").write_text(json.dumps(meta, indent=1))
    return res


def main() -> None:
    if (DIR / "results.json").exists():
        sys.exit("Round P already scored once.")
    results = {}
    results["low_muscle_mass"] = run("low_muscle_mass", pd.concat([muscle("G"), muscle("H")]), muscle("J"), "2011-12 + 2013-14", "2017-18", 20)
    nt = ntprobnp()
    results["heart_strain_ntprobnp"] = run("heart_strain_ntprobnp", nt[0], nt[1], "1999-2002", "2003-04", 20) if nt else None
    results["low_testosterone"] = run("low_testosterone", pd.concat([testosterone("G"), testosterone("H")]), testosterone("I"), "2011-12 + 2013-14", "2015-16", 20)
    (DIR / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
