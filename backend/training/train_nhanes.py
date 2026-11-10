"""Lab-checked screeners trained by the Curanex team on the US CDC NHANES survey, where the truth is measured, not reported.

Each model is trained on one NHANES cycle and tested once on a later cycle (different people, different years):
- fatty_liver: liver fat by FibroScan (CAP >= 285 dB/m) from routine blood tests and body size. Train 2017-18, test 2021-23.
- liver_fibrosis: liver stiffness by FibroScan (>= 8.0 kPa, valid scans only) from routine blood tests. Train 2017-18, test 2021-23.
- iron_deficiency: ferritin < 15 ng/mL from a complete blood count. Train 2017-18, test 2021-23.
- osteoporosis: femoral neck DXA T-score <= -2.5 (NHANES III young-women reference), age 50+. Train 2013-14, test 2017-18.
- kidney_disease_lab: eGFR < 60 (CKD-EPI 2021) or urine albumin/creatinine >= 30 mg/g, from questions only. Train 2017-18, test 2021-23.
- prediabetes: HbA1c >= 5.7% in people never told they have diabetes, from questions only. Train 2017-18, test 2021-23.
- high_bp_measured: average of 3 readings >= 140/90, from questions only. Train 2017-18, test 2021-23.
- high_cholesterol_measured: total cholesterol >= 240 mg/dL, from questions only. Train 2017-18, test 2021-23.

Rules fixed before looking at the test cycles: 80/20 stratified train/validation; logistic regression unless boosting wins
by 0.02 validation AUC; threshold highest with validation sensitivity >= 0.80; test cycle scored once; passes only with
test sensitivity >= 0.80 and AUC >= 0.80. Race is not used. Run from backend/: python training/train_nhanes.py [target ...]
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
DATA = BACKEND / "data" / "eval" / "nhanes"
OUT = BACKEND / "workers" / "tabular" / "models" / "nhanes"
CYCLE_YEARS = {"H": "2013-14", "J": "2017-18", "L": "2021-23"}
FN_BMD_YOUNG_MEAN, FN_BMD_YOUNG_SD = 0.858, 0.120  # NHANES III, women 20-29 (Looker et al. 1998)


def xpt(name: str, cycle: str, cols: list[str]) -> pd.DataFrame:
    path = DATA / f"{name}_{cycle}.xpt"
    if not path.exists():
        return pd.DataFrame(columns=["SEQN", *cols])
    d = pd.read_sas(path, format="xport")
    return d[["SEQN", *[c for c in cols if c in d.columns]]]


def yes(s: pd.Series) -> pd.Series:
    return s.map({1.0: 1.0, 2.0: 0.0})


def egfr(cr: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    """CKD-EPI 2021 creatinine equation (no race term)."""
    k = np.where(female == 1, 0.7, 0.9)
    a = np.where(female == 1, -0.241, -0.302)
    r = cr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938**age * np.where(female == 1, 1.012, 1.0)


def cycle(c: str) -> pd.DataFrame:
    d = xpt("DEMO", c, ["RIAGENDR", "RIDAGEYR"])
    for name, cols in (
        ("BMX", ["BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]),
        ("BIOPRO", ["LBXSATSI", "LBXSASSI", "LBXSGTSI", "LBXSAL", "LBXSTR", "LBXSCR", "LBXSGL"]),
        ("CBC", ["LBXHGB", "LBXMCVSI", "LBXMCHSI", "LBXRDW", "LBXPLTSI"]),
        ("GHB", ["LBXGH"]),
        ("LUX", ["LUXCAPM", "LUXSMED", "LUXSIQRM", "LUAXSTAT"]),
        ("FERTIN", ["LBXFER"]),
        ("ALB_CR", ["URDACT"]),
        ("DIQ", ["DIQ010"]),
        ("SMQ", ["SMQ020", "SMQ040"]),
        ("BPQ", ["BPQ020"]),
        ("MCQ", ["MCQ160C", "MCQ160F", "MCQ160B"]),
        ("DXXFEM", ["DXXNKBMD"]),
        ("BPXO", ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"]),
        ("TCHOL", ["LBXTC"]),
    ):
        d = d.merge(xpt(name, c, cols), on="SEQN", how="left")
    f = pd.DataFrame({"seqn": d["SEQN"]})
    f["age"] = d["RIDAGEYR"]
    f["male"] = d["RIAGENDR"].map({1.0: 1.0, 2.0: 0.0})
    f["weight_kg"], f["height_cm"], f["bmi"], f["waist_cm"] = d.get("BMXWT"), d.get("BMXHT"), d.get("BMXBMI"), d.get("BMXWAIST")
    for k, col in (
        ("alt", "LBXSATSI"),
        ("ast", "LBXSASSI"),
        ("ggt", "LBXSGTSI"),
        ("albumin", "LBXSAL"),
        ("triglycerides", "LBXSTR"),
        ("hemoglobin", "LBXHGB"),
        ("mcv", "LBXMCVSI"),
        ("mch", "LBXMCHSI"),
        ("rdw", "LBXRDW"),
        ("platelets", "LBXPLTSI"),
        ("hba1c", "LBXGH"),
        ("glucose", "LBXSGL"),
    ):
        f[k] = d.get(col)
    f["diabetes"] = d["DIQ010"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0}) if "DIQ010" in d else np.nan
    f["smoker_current"] = np.where(d.get("SMQ020") == 2, 0.0, d.get("SMQ040").map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0}))
    f["smoker_ever"] = yes(d["SMQ020"]) if "SMQ020" in d else np.nan
    f["high_bp"] = yes(d["BPQ020"]) if "BPQ020" in d else np.nan
    f["heart_or_stroke"] = np.fmax(
        np.fmax(yes(d.get("MCQ160C", pd.Series(np.nan, index=d.index))), yes(d.get("MCQ160F", pd.Series(np.nan, index=d.index)))),
        yes(d.get("MCQ160B", pd.Series(np.nan, index=d.index))),
    )
    # Targets
    valid = d.get("LUAXSTAT") == 1
    f["fatty_liver"] = np.where(valid & d["LUXCAPM"].notna(), (d["LUXCAPM"] >= 285).astype(float), np.nan) if "LUXCAPM" in d else np.nan
    if "LUXSMED" in d:
        ok = valid & d["LUXSMED"].notna() & (d["LUXSIQRM"] <= 30)
        f["liver_fibrosis"] = np.where(ok, (d["LUXSMED"] >= 8.0).astype(float), np.nan)
    f["iron_deficiency"] = np.where(d["LBXFER"].notna(), (d["LBXFER"] < 15).astype(float), np.nan) if "LBXFER" in d else np.nan
    if "DXXNKBMD" in d:
        t = (d["DXXNKBMD"] - FN_BMD_YOUNG_MEAN) / FN_BMD_YOUNG_SD
        f["osteoporosis"] = np.where(d["DXXNKBMD"].notna() & (d["RIDAGEYR"] >= 50), (t <= -2.5).astype(float), np.nan)
    if "LBXSCR" in d:
        e = egfr(d["LBXSCR"], d["RIDAGEYR"], (d["RIAGENDR"] == 2).astype(int))
        known = d["LBXSCR"].notna() & d.get("URDACT").notna() & (d["RIDAGEYR"] >= 20)
        f["kidney_disease_lab"] = np.where(known, ((e < 60) | (d["URDACT"] >= 30)).astype(float), np.nan)
    if "LBXGH" in d:
        f["prediabetes"] = np.where(
            d["LBXGH"].notna() & (f["diabetes"] == 0) & (d["RIDAGEYR"] >= 18), (d["LBXGH"] >= 5.7).astype(float), np.nan
        )
    if "BPXOSY1" in d:
        sy = d[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
        di = d[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
        f["high_bp_measured"] = np.where(sy.notna() & di.notna(), ((sy >= 140) | (di >= 90)).astype(float), np.nan)
    if "LBXTC" in d:
        f["high_cholesterol_measured"] = np.where(d["LBXTC"].notna(), (d["LBXTC"] >= 240).astype(float), np.nan)
    return f


TARGETS = {
    "fatty_liver": ("J", "L", ["age", "male", "bmi", "waist_cm", "alt", "ast", "ggt", "triglycerides", "hba1c", "diabetes"], 18),
    "liver_fibrosis": ("J", "L", ["age", "male", "bmi", "ast", "alt", "platelets", "albumin", "ggt", "diabetes"], 18),
    "iron_deficiency": ("J", "L", ["age", "male", "hemoglobin", "mcv", "mch", "rdw", "platelets"], 12),
    "osteoporosis": ("H", "J", ["age", "male", "weight_kg", "height_cm", "smoker_current", "smoker_ever", "diabetes"], 50),
    "kidney_disease_lab": ("J", "L", ["age", "male", "bmi", "diabetes", "high_bp", "heart_or_stroke", "smoker_current"], 20),
    "prediabetes": ("J", "L", ["age", "male", "bmi", "waist_cm", "high_bp", "heart_or_stroke", "smoker_current"], 18),
    # batch 2 (2026-11-10): measured, from questions only; being told of high BP is not an input
    "high_bp_measured": ("J", "L", ["age", "male", "bmi", "waist_cm", "diabetes", "heart_or_stroke", "smoker_current"], 18),
    "high_cholesterol_measured": ("J", "L", ["age", "male", "bmi", "waist_cm", "diabetes", "high_bp", "smoker_current"], 20),
}


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def gb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
        "specificity": round(float(tn / max(tn + fp, 1)), 3),
        "auc": round(float(roc_auc_score(y, p)), 3),
    }


def train(target: str, data: dict[str, pd.DataFrame]) -> dict:
    src, dst, feats, min_age = TARGETS[target]
    a, b = data[src], data[dst]
    a = a[a[target].notna() & (a["age"] >= min_age)]
    b = b[b[target].notna() & (b["age"] >= min_age)]
    x, y = a[feats].to_numpy(np.float32), a[target].to_numpy(int)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    fitted, scores = {}, {}
    for name, make in (("logistic", lr), ("boosting", gb)):
        fitted[name] = make().fit(x[tr], y[tr])
        scores[name] = roc_auc_score(y[va], fitted[name].predict_proba(x[va])[:, 1])
    name = "boosting" if scores["boosting"] > scores["logistic"] + 0.02 else "logistic"
    model = fitted[name]
    pv = model.predict_proba(x[va])[:, 1]
    t = float(max(v for v in np.unique(np.round(pv, 5)) if (pv[y[va] == 1] >= v).mean() >= 0.80))
    test = report(b[target].to_numpy(int), model.predict_proba(b[feats].to_numpy(np.float32))[:, 1], t)
    passed = test["sensitivity"] >= 0.80 and test["auc"] >= 0.80
    print(
        f"{target:18s} {name:8s} train {len(y)} ({int(y.sum())}+) val AUC lr {scores['logistic']:.3f} gb {scores['boosting']:.3f} | "
        f"test {CYCLE_YEARS[dst]} {test} {'PASS' if passed else 'fail'}",
        flush=True,
    )
    meta = {
        "target": target,
        "features": feats,
        "model": name,
        "threshold": t,
        "passed": passed,
        "min_age": min_age,
        "metrics": {"validation_auc": round(scores[name], 3), "test": test},
        "trained_on": f"US CDC NHANES {CYCLE_YEARS[src]}, {len(tr):,} people",
        "test_set": f"NHANES {CYCLE_YEARS[dst]}, different people",
        "median": np.nanmedian(x[tr], axis=0).tolist(),
        "train_min": np.nanmin(x[tr], axis=0).tolist(),
        "train_max": np.nanmax(x[tr], axis=0).tolist(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if name == "logistic":
        imp, sc, clf = (model.steps[i][1] for i in range(3))
        meta |= {
            "impute": imp.statistics_.tolist(),
            "mean": sc.mean_.tolist(),
            "scale": sc.scale_.tolist(),
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
        }
    else:
        from skl2onnx import to_onnx

        onx = to_onnx(model, x[:1], options={id(model): {"zipmap": False}}, target_opset=17)
        (OUT / f"{target}.onnx").write_bytes(onx.SerializeToString())
        meta["onnx_sha256"] = hashlib.sha256((OUT / f"{target}.onnx").read_bytes()).hexdigest()
    (OUT / f"{target}.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    data = {c: cycle(c) for c in CYCLE_YEARS}
    for target in sys.argv[1:] or list(TARGETS):
        train(target, data)


if __name__ == "__main__":
    main()
