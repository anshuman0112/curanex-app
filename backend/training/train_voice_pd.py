"""Parkinson's voice screening on a sustained "aaah". Trained by the Curanex team, following the idea of
mohnabilfadl/Parkinson_Detection_Using_Voice_Analysis, on the UCI "Parkinson's Disease Classification" set
(Sakar et al. 2018: 252 people, 3 recordings each, Istanbul). Run from backend/.

The repo's model uses RPDE, D2 and PPE, which need custom research code. This model only uses values that Praat
(praat-parselmouth) computes from a recording, so the app can compute exactly the same inputs. Loudness is left out
because it depends on the phone and the distance.

Rules fixed before looking at the test split:
- split by person (60/20/20, stratified), so no person's voice is in two splits
- log of jitter, shimmer, noise ratio and period spread (they are skewed)
- logistic regression unless boosting wins by 0.02 AUC in training folds (it did: 0.774 against 0.736), exported to ONNX
- threshold: highest with validation sensitivity >= 0.85
- test split scored once, per recording and per person (mean of the person's recordings)
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "eval" / "pd_sakar" / "pd_speech_features.csv"
OUT = BACKEND / "api" / "voice_models" / "parkinson_voice.json"
ONNX = OUT.with_suffix(".onnx")
FEATURES = ["male", "meanPeriodPulses", "stdDevPeriodPulses", "locPctJitter", "locAbsJitter", "rapJitter", "ppq5Jitter", "ddpJitter",
            "locShimmer", "locDbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "ddaShimmer",
            "meanAutoCorrHarmonicity", "meanNoiseToHarmHarmonicity", "meanHarmToNoiseHarmonicity", "f1", "f2", "f3", "f4"]
LOG = {"stdDevPeriodPulses", "locPctJitter", "locAbsJitter", "rapJitter", "ppq5Jitter", "ddpJitter", "locShimmer", "locDbShimmer",
       "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "ddaShimmer", "meanNoiseToHarmHarmonicity"}
TARGET_SENSITIVITY = 0.85
EPS = 1e-9


def load():
    with DATA.open(encoding="utf-8") as f:
        f.readline()
        rows = list(csv.DictReader(f))
    x = np.array([[float(r["gender"]) if c == "male" else float(r[c]) for c in FEATURES] for r in rows])
    for i, c in enumerate(FEATURES):
        if c in LOG:
            x[:, i] = np.log(np.maximum(x[:, i], EPS))
    return x, np.array([int(r["class"]) for r in rows]), np.array([int(r["id"]) for r in rows])


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))


def gb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=42)


def threshold_for(y, p):
    return float(max(t for t in np.unique(p) if (p[y == 1] >= t).mean() >= TARGET_SENSITIVITY))


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, p)), 3)}


def per_person(y, p, g):
    ids = np.unique(g)
    return np.array([y[g == i][0] for i in ids]), np.array([p[g == i].mean() for i in ids])


def main() -> None:
    x, y, g = load()
    people = np.unique(g)
    label = np.array([y[g == i][0] for i in people])
    print(f"{len(y)} recordings, {len(people)} people, {int(label.sum())} with Parkinson's", flush=True)
    tr_p, rest = train_test_split(people, test_size=0.40, random_state=42, stratify=label)
    va_p, te_p = train_test_split(rest, test_size=0.50, random_state=42, stratify=label[np.isin(people, rest)])
    tr, va, te = np.isin(g, tr_p), np.isin(g, va_p), np.isin(g, te_p)

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x[tr], y[tr], g[tr]))
    a_lr = roc_auc_score(y[tr], cross_val_predict(lr(), x[tr], y[tr], cv=folds, method="predict_proba")[:, 1])
    a_gb = roc_auc_score(y[tr], cross_val_predict(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=42),
                                                  x[tr], y[tr], cv=folds, method="predict_proba")[:, 1])
    print(f"training-fold AUC: logistic {a_lr:.3f}, boosting {a_gb:.3f}", flush=True)
    make = gb if a_gb > a_lr + 0.02 else lr
    model = make().fit(x[tr], y[tr])
    t = threshold_for(y[va], model.predict_proba(x[va])[:, 1])
    pv = model.predict_proba(x[te])[:, 1]
    metrics = {"test_recordings": report(y[te], pv, t), "test_people": report(*per_person(y[te], pv, g[te]), t), "threshold": round(t, 3)}
    print("test (once):", json.dumps(metrics), flush=True)

    # The threshold was set on validation with the model trained on the training split, so that model is the one exported.
    from skl2onnx import to_onnx
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onx = to_onnx(model, x[:1].astype(np.float32), options={id(model): {"zipmap": False}} if make is gb else None, target_opset=17)
    ONNX.write_bytes(onx.SerializeToString())
    import onnxruntime as ort
    s = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    diff = float(np.abs(s.run(None, {s.get_inputs()[0].name: x[te].astype(np.float32)})[1][:, 1] - pv).max())
    print("onnx max diff", diff, flush=True)
    assert diff < 1e-4
    import hashlib
    OUT.write_text(json.dumps({
        "features": FEATURES, "log": sorted(LOG), "onnx_sha256": hashlib.sha256(ONNX.read_bytes()).hexdigest(),
        "input": s.get_inputs()[0].name, "train_mean": x[tr].mean(0).tolist(), "train_std": (x[tr].std(0) + EPS).tolist(), "model": make.__name__, "threshold": t, "metrics": metrics,
        "training_people": int(len(tr_p)),
        "trained_on": "UCI Parkinson's Disease Classification (Sakar et al. 2018): 252 people in Istanbul, studio microphone, sustained 'a'",
    }, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
