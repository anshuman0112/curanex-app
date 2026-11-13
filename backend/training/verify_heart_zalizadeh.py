"""Outside test for the heart disease risk check on Z-Alizadeh Sani (UCI 412: 303 patients, Tehran, Iran,
coronary disease confirmed by angiography). Run from backend/.

Rules fixed before the Iranian data is scored (written 2026-11-13):
1. `choose`: pick the algorithm on the UCI four-hospital data only (each hospital held out in turn).
   Candidates: logistic regression (the live model), histogram gradient boosting, random forest.
   The live logistic model is kept unless another candidate beats its mean held-out AUC by more than 0.02
   AND its worst-hospital AUC is not lower.
2. `score`: the chosen model is scored ONCE on Z-Alizadeh Sani, at the threshold already stored in the
   model file (no retuning). Pass = sensitivity >= 0.80 and AUC >= 0.80.
   Input mapping, same as a patient filling the app form:
   - chest pain: Typical -> "Pressure on effort", Atypical -> "Unusual chest discomfort",
     Nonanginal -> "Chest pain that doesn't feel heart related", none of those -> "No chest pain"
   - pain on exertion (required in the app): Yes for typical chest pain, otherwise No
     (the dataset's own "Exertional CP" column is "N" for all 303 rows, so it cannot be used)
   - cholesterol: total cholesterol by the Friedewald identity LDL + HDL + TG/5 (mg/dL)
   - high fasting sugar: FBS > 120 mg/dL; resting BP: BP column; sex and age as given
   - label: Cath == "Cad"
3. The result is written to data/eval/zalizadeh/result.json; `score` refuses to run a second time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_heart import SITES, calibrated, load, lr  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "eval" / "zalizadeh" / "Z-Alizadeh sani dataset.xlsx"
RESULT = BACKEND / "data" / "eval" / "zalizadeh" / "result.json"
META = BACKEND / "workers" / "tabular" / "models" / "sneha798_heart_disease_prediction_project" / "model_meta.json"


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3),
            "sensitivity": round(float(tp / max(tp + fn, 1)), 3), "specificity": round(float(tn / max(tn + fp, 1)), 3),
            "auc": round(float(roc_auc_score(y, p)), 3)}


def choose() -> None:
    x, y, site = load()
    makers = {
        "logistic": lr,
        "boosting": lambda: make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=42)),
        "forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(400, min_samples_leaf=5, random_state=42, n_jobs=-1)),
    }
    out = {}
    for name, make in makers.items():
        aucs = {}
        for s in SITES:
            tr, te = site != s, site == s
            aucs[s] = round(float(roc_auc_score(y[te], make().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1])), 3)
        out[name] = {"each": aucs, "mean": round(float(np.mean(list(aucs.values()))), 3), "worst": min(aucs.values())}
        print(name, out[name], flush=True)
    best = "logistic"
    for name in ("boosting", "forest"):
        if out[name]["mean"] > out["logistic"]["mean"] + 0.02 and out[name]["worst"] >= out["logistic"]["worst"]:
            best = name if best == "logistic" or out[name]["mean"] > out[best]["mean"] else best
    print("chosen:", best)


def outside():
    d = pd.read_excel(DATA)
    typical = d["Typical Chest Pain"] == 1
    atyp = (d["Atypical"] == "Y") & ~typical
    nonang = (d["Nonanginal"] == "Y") & ~typical & ~atyp
    x = pd.DataFrame({
        "age": d["Age"], "male": (d["Sex"] == "Male").astype(float),
        "pain_typical": typical.astype(float), "pain_atypical": atyp.astype(float), "pain_other": nonang.astype(float),
        "pain_none": (~typical & ~atyp & ~nonang).astype(float),
        "resting_bp": d["BP"].astype(float), "cholesterol": d["LDL"] + d["HDL"] + d["TG"] / 5,
        "high_fasting_sugar": (d["FBS"] > 120).astype(float), "pain_on_exertion": typical.astype(float),
    })
    return x, (d["Cath"] == "Cad").astype(int).to_numpy()


def score() -> None:
    if RESULT.exists():
        sys.exit(f"Already scored once: {RESULT}")
    m = json.loads(META.read_text())
    x, y = outside()
    x = x[m["features"]].to_numpy(float)
    d = ((x - np.array(m["mean"])) / np.array(m["scale"]) * np.array(m["coef"])).sum(1) + m["intercept"]
    a, b = m["platt"]
    p = 1 / (1 + np.exp(-(a * d + b)))
    res = report(y, p, m["threshold"])
    res |= {"passed": res["sensitivity"] >= 0.80 and res["auc"] >= 0.80, "model": "live logistic model, unchanged",
            "set": "Z-Alizadeh Sani (UCI 412), Tehran, Iran, angiography-confirmed", "scored": "2026-11-13"}
    RESULT.write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    {"choose": choose, "score": score}[sys.argv[1]]()
