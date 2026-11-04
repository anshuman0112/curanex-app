"""Heart disease check on questions a patient can answer. Trained by the Curanex team using the approach of
sneha798/Heart-Disease-Prediction-Project, on the UCI Heart Disease data from four hospitals. Run from backend/.

The repo's model needs angiography and stress-test values, so those columns are left out on purpose.
Rules fixed before looking at the test split: logistic regression unless boosting wins by 0.02 AUC in training folds;
threshold is the highest with validation sensitivity >= 0.85; the test split is scored once; then each hospital is
held out in turn, which is the harder and more honest number.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "eval" / "heart"
OUT = BACKEND / "workers" / "tabular" / "models" / "sneha798_heart_disease_prediction_project"
SITES = {"cleveland": "Cleveland (USA)", "hungarian": "Budapest (Hungary)", "switzerland": "Switzerland", "va": "Long Beach VA (USA)"}
FEATURES = ["age", "male", "pain_typical", "pain_atypical", "pain_other", "pain_none", "resting_bp", "cholesterol", "high_fasting_sugar", "pain_on_exertion"]
TARGET_SENSITIVITY = 0.85


def load():
    rows, y, site = [], [], []
    for name in SITES:
        for line in (DATA / f"processed.{name}.data").read_text().split("\n"):
            parts = line.strip().split(",")
            if len(parts) != 14:
                continue
            v = [np.nan if p == "?" else float(p) for p in parts]
            age, sex, cp, bp, chol, fbs, exang = v[0], v[1], v[2], v[3], v[4], v[5], v[8]
            rows.append([age, sex, float(cp == 1), float(cp == 2), float(cp == 3), float(cp == 4), bp or np.nan, chol or np.nan, fbs, exang])
            y.append(int(v[13] > 0))
            site.append(name)
    return np.array(rows), np.array(y), np.array(site)


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def calibrated(x, y):
    return CalibratedClassifierCV(lr(), method="sigmoid", cv=5).fit(x, y)


def threshold_for(y, p):
    return float(max(t for t in np.unique(p) if (p[y == 1] >= t).mean() >= TARGET_SENSITIVITY))


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, p)), 3)}


def main() -> None:
    x, y, site = load()
    print({s: (int((site == s).sum()), round(float(y[site == s].mean()), 2)) for s in SITES}, flush=True)
    strat = np.array([f"{s}{t}" for s, t in zip(site, y, strict=True)])
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.30, random_state=42, stratify=strat)
    val, test = train_test_split(rest, test_size=0.50, random_state=42, stratify=strat[rest])
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    a_lr = roc_auc_score(y[train], cross_val_predict(lr(), x[train], y[train], cv=folds, method="predict_proba")[:, 1])
    a_gb = roc_auc_score(y[train], cross_val_predict(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=42), x[train], y[train], cv=folds, method="predict_proba")[:, 1])
    print(f"training-fold AUC: logistic {a_lr:.3f}, boosting {a_gb:.3f}", flush=True)
    if a_gb > a_lr + 0.02:
        sys.exit("Boosting won by more than 0.02: stop and export it instead of the logistic model. Test split left untouched.")

    model = calibrated(x[train], y[train])
    t = threshold_for(y[val], model.predict_proba(x[val])[:, 1])
    metrics = {"test": report(y[test], model.predict_proba(x[test])[:, 1], t)}
    print("test (once):", metrics["test"], "threshold", round(t, 3), flush=True)

    holdout = {}
    for s, title in SITES.items():
        tr, te = site != s, site == s
        oof = cross_val_predict(lr(), x[tr], y[tr], cv=folds, method="predict_proba")[:, 1]
        m = calibrated(x[tr], y[tr])
        holdout[title] = report(y[te], m.predict_proba(x[te])[:, 1], threshold_for(y[tr], oof))
        print("held-out hospital", title, holdout[title], flush=True)
    metrics["each_hospital_held_out"] = holdout

    # Final model on train + val, as plain numbers: nothing to unpickle at run time.
    final = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(x[np.r_[train, val]], y[np.r_[train, val]])
    imp, sc, clf = final.steps[0][1], final.steps[1][1], final.steps[2][1]
    raw = final.decision_function(x[val])
    platt = LogisticRegression(C=1e6).fit(raw.reshape(-1, 1), y[val])
    p = lambda d: 1 / (1 + np.exp(-(platt.coef_[0, 0] * d + platt.intercept_[0])))  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_meta.json").write_text(json.dumps({
        "features": FEATURES, "median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "platt": [float(platt.coef_[0, 0]), float(platt.intercept_[0])],
        "threshold": threshold_for(y[val], p(raw)), "metrics": metrics, "training_rows": len(train) + len(val),
    }, indent=1))
    print("wrote", OUT / "model_meta.json")


if __name__ == "__main__":
    main()
