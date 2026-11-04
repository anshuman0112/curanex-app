"""Thyroid pattern check from a thyroid lab report. Trained by the Curanex team using the method of
cilab-ufersa/thyroid_disease_AI, on the UCI Garavan thyroid data, without the repo's leakage (it imputed from the label and
applied SMOTE before splitting). Run from backend/.

Rules fixed before scoring: the official .test file is the test set, scored once; 20% of .data is validation; threshold is
the highest with validation sensitivity >= 0.85; boosting is used only if it beats logistic by 0.02 AUC in training folds.
Stress check: every lab value except TSH removed, the usual case when someone has one test.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "eval" / "thyroid"
OUT = BACKEND / "workers" / "tabular" / "models" / "cilab_ufersa_thyroid_disease_ai"
FEATURES = ["age", "female", "on_thyroxine", "on_antithyroid", "log_tsh", "t3", "tt4", "t4u", "fti"]
LABS = ["t3", "tt4", "t4u", "fti"]
TARGET_SENSITIVITY = 0.85


def num(v):
    return np.nan if v == "?" else float(v)


def read(suffix):
    hyper = {}
    for line in (DATA / f"allhyper.{suffix}").read_text().split("\n"):
        if "|" in line:
            body, rid = line.strip().split("|")
            hyper[rid] = body.split(",")[-1].rstrip(".").strip()
    rows, hypo_y, hyper_y = [], [], []
    for line in (DATA / f"allhypo.{suffix}").read_text().split("\n"):
        if "|" not in line:
            continue
        body, rid = line.strip().split("|")
        v = body.split(",")
        age = num(v[0])
        if age > 110:  # a typing error in the source data
            age = np.nan
        tsh = num(v[17])
        rows.append([age, np.nan if v[1] == "?" else float(v[1] == "F"), float(v[2] == "t"), float(v[4] == "t"),
                     np.log1p(tsh) if tsh == tsh else np.nan, num(v[19]), num(v[21]), num(v[23]), num(v[25])])
        hypo_y.append(int("hypothyroid" in v[-1]))
        hyper_y.append(int(hyper.get(rid, "negative") != "negative"))
    return np.array(rows), np.array(hypo_y), np.array(hyper_y)


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))


def gb():
    return make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.08, random_state=42))


def threshold_for(y, p):
    return float(max(t for t in np.unique(p) if (p[y == 1] >= t).mean() >= TARGET_SENSITIVITY))


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, p)), 3)}


def export_trees(pipe):
    imp, gbc = pipe.steps[0][1], pipe.steps[1][1]
    trees = [{"left": t.tree_.children_left.tolist(), "right": t.tree_.children_right.tolist(), "feature": t.tree_.feature.tolist(),
              "threshold": [round(float(x), 6) for x in t.tree_.threshold], "value": [round(float(x), 6) for x in t.tree_.value.ravel()]} for t in gbc.estimators_.ravel()]
    prior = float(np.log(gbc.init_.class_prior_[1] / gbc.init_.class_prior_[0]))
    return {"median": imp.statistics_.tolist(), "prior": prior, "rate": gbc.learning_rate, "trees": trees}


def main() -> None:
    x, hypo, hyper = read("data")
    xt, hypot, hypert = read("test")
    print(len(x), int(hypo.sum()), int(hyper.sum()), "| test", len(xt), int(hypot.sum()), int(hypert.sum()), flush=True)
    meta = {"features": FEATURES, "training_rows": len(x), "metrics": {}, "models": {}}
    no_labs = xt.copy()
    no_labs[:, [FEATURES.index(k) for k in LABS]] = np.nan
    for name, y, yt in (("underactive", hypo, hypot), ("overactive", hyper, hypert)):
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
        folds = StratifiedKFold(5, shuffle=True, random_state=42)
        a_lr = roc_auc_score(y[tr], cross_val_predict(lr(), x[tr], y[tr], cv=folds, method="predict_proba")[:, 1])
        a_gb = roc_auc_score(y[tr], cross_val_predict(gb(), x[tr], y[tr], cv=folds, method="predict_proba")[:, 1])
        print(name, f"training-fold AUC: logistic {a_lr:.3f}, boosting {a_gb:.3f}", flush=True)
        make = gb if a_gb > a_lr + 0.02 else lr
        model = make().fit(x[tr], y[tr])
        t = threshold_for(y[va], model.predict_proba(x[va])[:, 1])
        result = {"model": make.__name__, "test": report(yt, model.predict_proba(xt)[:, 1], t), "only_tsh": report(yt, model.predict_proba(no_labs)[:, 1], t)}
        print(name, result, flush=True)
        meta["metrics"][name] = result
        if make is gb:
            meta["models"][name] = {"kind": "trees", "threshold": t, **export_trees(model)}
        else:
            imp, sc, clf = model.steps[0][1], model.steps[1][1], model.steps[2][1]
            meta["models"][name] = {"kind": "logistic", "threshold": t, "median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                                    "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_meta.json").write_text(json.dumps(meta))
    print("wrote", (OUT / "model_meta.json").stat().st_size)


if __name__ == "__main__":
    main()
