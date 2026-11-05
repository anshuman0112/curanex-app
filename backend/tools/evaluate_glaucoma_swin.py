"""Tests pamixsun's Swin glaucoma classifier on the independent retina set: python tools/evaluate_glaucoma_swin.py (from backend/).
Half (seed 42) sets the threshold, highest with sensitivity >= 0.85; the other half is scored once, against normal eyes and,
harder, against normal eyes plus the other eye diseases.
"""
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND / "data" / "eval" / "cataract" / "dataset"
MODEL = BACKEND / "workers" / "torch" / "models" / "pamixsun_glaucoma"
FOLDERS = {"normal": "1_normal", "cataract": "2_cataract", "glaucoma": "2_glaucoma", "retina": "3_retina_disease"}
MEAN, STD = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
session = None


def init() -> None:
    global session
    o = ort.SessionOptions()
    o.intra_op_num_threads = 2
    session = ort.InferenceSession(str(MODEL / "model.onnx"), o, providers=["CPUExecutionProvider"])


def one(path: str):
    try:
        a = np.asarray(Image.open(path).convert("RGB").resize((512, 512), Image.BICUBIC), dtype=np.float32) / 255
    except OSError:
        return None
    x = ((a - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]
    logits = session.run(None, {"image": x})[0][0]
    e = np.exp(logits - logits.max())
    return float(e[1] / e.sum())


def auc(y, s):
    r = np.empty(len(s))
    r[s.argsort()] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        r[s == v] = r[s == v].mean()
    pos, neg = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def report(y, s, t):
    p = s >= t
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / max(tp + fn, 1), 3),
            "specificity": round(tn / max(tn + fp, 1), 3), "auc": round(auc(y, s), 3)}


def main() -> None:
    items = [(str(p), k) for k, f in FOLDERS.items() for p in sorted((ROOT / f).glob("*"))]
    with Pool(2, initializer=init) as pool:
        rows = pool.map(one, [p for p, _ in items], chunksize=4)
    keep = [i for i, r in enumerate(rows) if r is not None]
    s = np.array([rows[i] for i in keep])
    kind = np.array([items[i][1] for i in keep])
    idx = list(range(len(kind)))
    random.Random(42).shuffle(idx)  # noqa: S311 sampling, not security
    va, te = np.array(sorted(idx[: len(idx) // 2])), np.array(sorted(idx[len(idx) // 2 :]))
    y = (kind == "glaucoma").astype(int)
    m = np.isin(kind[va], ["glaucoma", "normal"])
    print("half-1 AUC", round(auc(y[va][m], s[va][m]), 3), flush=True)
    t = float(max(v for v in np.unique(s[va][y[va] == 1]) if (s[va][y[va] == 1] >= v).mean() >= 0.85))
    easy = np.isin(kind[te], ["glaucoma", "normal"])
    result = {"threshold": t, "test_vs_normal": report(y[te][easy], s[te][easy], t), "test_vs_everything_else": report(y[te], s[te], t)}
    print(json.dumps(result), flush=True)
    (MODEL / "evaluation.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
