"""Risk screeners from questions a person can answer, trained by the Curanex team on the US CDC BRFSS survey.

BRFSS asks about 440,000 adults a year whether a doctor ever told them they have a condition, plus age, sex, BMI,
smoking, alcohol, exercise, sleep and general health. Each model predicts one condition from the other answers.

Rules fixed before looking at the test year:
- train and validate on BRFSS 2022 (80/20, stratified); test once on BRFSS 2023, which is different people
  (blood pressure and cholesterol are asked only in odd years: trained on 2021 instead)
- sleep is left out: the 2023 survey did not ask it
- race, income, education and US state are left out: they are US-specific and not relevant for Indian users
- the condition itself and its near-duplicates are never inputs (for example heart attack for coronary disease)
- logistic regression unless boosting wins by 0.02 AUC on validation
- threshold: highest with validation sensitivity >= 0.80
- passes the Curanex bar only with 2023 sensitivity >= 0.80 and AUC >= 0.80
Run from backend/: python training/train_brfss.py [target ...]
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "eval" / "brfss"
OUT = BACKEND / "workers" / "tabular" / "models" / "survey"
RAW = [
    "SEXVAR",
    "_AGEG5YR",
    "_BMI5",
    "_SMOKER3",
    "USENOW3",
    "_RFDRHV8",
    "EXERANY2",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "SLEPTIM1",
    "DIFFWALK",
    "DECIDE",
    "_MICHD",
    "CVDSTRK3",
    "ASTHMA3",
    "CHCCOPD3",
    "CHCKDNY2",
    "HAVARTH4",
    "ADDEPEV3",
    "DIABETE4",
    "CHCOCNC1",
    "BPHIGH6",
    "TOLDHI3",
]
# Same questions under other names in 2021.
RENAME = {"_RFDRHV7": "_RFDRHV8", "HAVARTH5": "HAVARTH4", "CHCOCNCR": "CHCOCNC1"}
AGE_MID = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 82}
BASE = [
    "age",
    "male",
    "bmi",
    "smoker_current",
    "smoker_former",
    "smokeless_tobacco",
    "heavy_drinker",
    "exercise",
    "general_health",
    "poor_physical_days",
    "poor_mental_days",
    "difficulty_walking",
    "difficulty_concentrating",
]
CONDITIONS = ["diabetes", "heart_disease", "stroke", "asthma", "copd", "kidney_disease", "arthritis", "depression"]
# Inputs never used for a target, beyond the target itself.
EXCLUDE = {"copd": {"asthma"}, "asthma": {"copd"}}
TARGETS = {
    "heart_disease": "Coronary heart disease or heart attack",
    "stroke": "Stroke",
    "copd": "COPD (chronic lung disease)",
    "kidney_disease": "Chronic kidney disease",
    "arthritis": "Arthritis",
    "depression": "Depression",
    "asthma": "Asthma",
    "diabetes": "Type 2 diabetes",
    "cancer": "Cancer (any, except skin)",
}
# Asked only in odd years: trained on 2021, tested on 2023.
ODD_YEAR = {"high_blood_pressure": "High blood pressure", "high_cholesterol": "High cholesterol"}


def yes(s: pd.Series) -> pd.Series:
    return s.map({1.0: 1.0, 2.0: 0.0})


def extract(year: int) -> pd.DataFrame:
    cache = DATA / f"brfss{year}_v2.csv"
    if cache.exists():
        return pd.read_csv(cache)
    parts = [c.rename(columns=RENAME).reindex(columns=RAW) for c in pd.read_sas(DATA / f"LLCP{year}.XPT", format="xport", chunksize=50000)]
    r = pd.concat(parts, ignore_index=True)
    days = lambda s: s.mask(s == 88, 0).where(lambda v: v <= 30)  # noqa: E731 - 88 means none, 77/99 unknown
    d = pd.DataFrame(
        {
            "age": r["_AGEG5YR"].map(AGE_MID),
            "male": r["SEXVAR"].map({1.0: 1.0, 2.0: 0.0}),
            "bmi": r["_BMI5"] / 100,
            "smoker_current": r["_SMOKER3"].map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0, 4.0: 0.0}),
            "smoker_former": r["_SMOKER3"].map({1.0: 0.0, 2.0: 0.0, 3.0: 1.0, 4.0: 0.0}),
            "smokeless_tobacco": r["USENOW3"].map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0}),
            "heavy_drinker": r["_RFDRHV8"].map({1.0: 0.0, 2.0: 1.0}),
            "exercise": yes(r["EXERANY2"]),
            "general_health": r["GENHLTH"].where(r["GENHLTH"] <= 5),
            "poor_physical_days": days(r["PHYSHLTH"]),
            "poor_mental_days": days(r["MENTHLTH"]),
            "sleep_hours": r["SLEPTIM1"].where(r["SLEPTIM1"] <= 24),
            "difficulty_walking": yes(r["DIFFWALK"]),
            "difficulty_concentrating": yes(r["DECIDE"]),
            "diabetes": r["DIABETE4"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0}),
            "heart_disease": r["_MICHD"].map({1.0: 1.0, 2.0: 0.0}),
            "stroke": yes(r["CVDSTRK3"]),
            "asthma": yes(r["ASTHMA3"]),
            "copd": yes(r["CHCCOPD3"]),
            "kidney_disease": yes(r["CHCKDNY2"]),
            "arthritis": yes(r["HAVARTH4"]),
            "depression": yes(r["ADDEPEV3"]),
            "cancer": yes(r["CHCOCNC1"]),
            "high_blood_pressure": r["BPHIGH6"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0}),
            "high_cholesterol": yes(r["TOLDHI3"]),
        }
    )
    d.to_csv(cache, index=False)
    return d


def features_for(target: str) -> list[str]:
    return BASE + [c for c in CONDITIONS if c != target and c not in EXCLUDE.get(target, set())]


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def gb():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=200, random_state=42)


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
        "specificity": round(float(tn / max(tn + fp, 1)), 3),
        "auc": round(float(roc_auc_score(y, p)), 3),
    }


def train(target: str, d22: pd.DataFrame, d23: pd.DataFrame, year: int = 2022) -> dict:
    feats = features_for(target)
    a = d22[d22[target].notna()]
    b = d23[d23[target].notna()]
    x, y = a[feats].to_numpy(np.float32), a[target].to_numpy(int)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    scores = {}
    fitted = {}
    for name, make in (("logistic", lr), ("boosting", gb)):
        m = make().fit(x[tr], y[tr])
        fitted[name] = m
        scores[name] = roc_auc_score(y[va], m.predict_proba(x[va])[:, 1])
    name = "boosting" if scores["boosting"] > scores["logistic"] + 0.02 else "logistic"
    model = fitted[name]
    pv = model.predict_proba(x[va])[:, 1]
    t = float(max(v for v in np.unique(np.round(pv, 5)) if (pv[y[va] == 1] >= v).mean() >= 0.80))
    xt, yt = b[feats].to_numpy(np.float32), b[target].to_numpy(int)
    test = report(yt, model.predict_proba(xt)[:, 1], t)
    passed = test["sensitivity"] >= 0.80 and test["auc"] >= 0.80
    print(
        f"{target:15s} {name:8s} val AUC lr {scores['logistic']:.3f} gb {scores['boosting']:.3f} | 2023 test {test} {'PASS' if passed else 'fail'}",
        flush=True,
    )
    meta = {
        "target": target,
        "title": (TARGETS | ODD_YEAR)[target],
        "features": feats,
        "model": name,
        "threshold": t,
        "passed": passed,
        "metrics": {"validation_auc": round(scores[name], 3), "test_2023": test},
        "trained_on": f"US CDC BRFSS {year} survey, {len(tr):,} adults (validated on {len(va):,})",
        "test_set": "BRFSS 2023, different people",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if name == "logistic":
        imp, sc, clf = (model.steps[i][1] for i in range(3))
        meta |= {
            "median": imp.statistics_.tolist(),
            "mean": sc.mean_.tolist(),
            "scale": sc.scale_.tolist(),
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
        }
    else:
        from skl2onnx import to_onnx

        onx = to_onnx(model, x[:1], options={id(model): {"zipmap": False}}, target_opset=17)
        path = OUT / f"{target}.onnx"
        path.write_bytes(onx.SerializeToString())
        meta["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        meta["median"] = np.nanmedian(x[tr], axis=0).tolist()
    (OUT / f"{target}.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    d22, d23 = extract(2022), extract(2023)
    print(f"2022: {len(d22):,} people, 2023: {len(d23):,} people", flush=True)
    wanted = sys.argv[1:] or [*TARGETS, *ODD_YEAR]
    for target in wanted:
        if target in TARGETS:
            train(target, d22, d23)
    if any(t in ODD_YEAR for t in wanted):
        d21 = extract(2021)
        print(f"2021: {len(d21):,} people", flush=True)
        for target in wanted:
            if target in ODD_YEAR:
                train(target, d21, d23, year=2021)


if __name__ == "__main__":
    main()
