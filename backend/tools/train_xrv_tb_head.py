"""TB signs layer on TorchXRayVision features, trained by the Curanex team. Run from backend/: python tools/train_xrv_tb_head.py

Why only NLM images for training: in this Kaggle database most normal images come from RSNA and TB images from elsewhere, so a
mixed model could learn the source instead of TB. NLM (Shenzhen and Montgomery) supplies both classes from the same places.
Rules fixed before scoring: 70/15/15 split; C and threshold chosen on validation, threshold the highest with sensitivity >= 0.85;
test scored once. Stress checks, with that threshold: Belarus TB images (a new source) and RSNA normals (a new source).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import openpyxl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from tools.extract_fast import run  # noqa: E402

ROOT = BACKEND / "data" / "eval" / "tbk" / "TB_Chest_Radiography_Database"
CACHE = BACKEND / "data" / "eval" / "features_tb.npz"
OUT = BACKEND / "workers" / "torch" / "models" / "mlmed_torchxrayvision" / "tb_adult_head.json"
TARGET = 0.85
OUTSIDE_NORMALS = 400


def rows(kind):
    ws = openpyxl.load_workbook(ROOT / f"{kind}.metadata.xlsx", read_only=True).active
    return [(ROOT / kind / f"{r[0]}.png", str(r[3])) for r in list(ws.iter_rows(values_only=True))[1:]]


def features():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=False)
        return d["X"], d["y"], d["source"]
    normal, tb = rows("Normal"), rows("Tuberculosis")
    rsna = [p for p, u in normal if "rsna" in u]
    random.Random(42).shuffle(rsna)  # noqa: S311 sampling, not security
    items = [(p, 0, "nlm") for p, u in normal if "PMC4256233" in u] + [(p, 1, "nlm") for p, u in tb if "PMC4256233" in u]
    items += [(p, 1, "belarus") for p, u in tb if "tuberculosis.by" in u] + [(p, 0, "rsna") for p in rsna[:OUTSIDE_NORMALS]]
    out = run([str(i[0]) for i in items])
    keep = [i for i, r in enumerate(out) if r is not None]
    X, y, source = np.stack([out[i][0] for i in keep]), np.array([items[i][1] for i in keep]), np.array([items[i][2] for i in keep])
    np.savez(CACHE, X=X, y=y, source=source)
    return X, y, source


def report(y, s, t):
    tn, fp, fn, tp = confusion_matrix(y, s >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, s)), 3)}


def main() -> None:
    X, y, source = features()
    nlm = np.where(source == "nlm")[0]
    print("nlm", len(nlm), int(y[nlm].sum()), flush=True)
    tr, rest = train_test_split(nlm, test_size=0.30, random_state=42, stratify=y[nlm])
    va, te = train_test_split(rest, test_size=0.50, random_state=42, stratify=y[rest])
    scaler = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=c, max_iter=3000).fit(scaler.transform(X[tr]), y[tr])
        a = roc_auc_score(y[va], m.predict_proba(scaler.transform(X[va]))[:, 1])
        print(f"C={c} validation AUC {a:.4f}", flush=True)
        if best is None or a > best[0]:
            best = (a, c, m)
    _, c, model = best
    score = lambda Z: model.predict_proba(scaler.transform(Z))[:, 1]  # noqa: E731
    s_va = score(X[va])
    t = float(max(v for v in np.unique(s_va) if (s_va[y[va] == 1] >= v).mean() >= TARGET))
    bel, rsna = source == "belarus", source == "rsna"
    both = bel | rsna
    results = {"C": c, "threshold": t, "test_nlm": report(y[te], score(X[te]), t), "new_sources_belarus_tb_vs_rsna_normal": report(y[both], score(X[both]), t),
               "belarus_tb_found": round(float((score(X[bel]) >= t).mean()), 3), "rsna_normal_cleared": round(float((score(X[rsna]) < t).mean()), 3)}
    print(json.dumps(results, indent=1), flush=True)
    OUT.write_text(json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist(),
                               "intercept": float(model.intercept_[0]), "threshold": t, "metrics": results}))


if __name__ == "__main__":
    main()
