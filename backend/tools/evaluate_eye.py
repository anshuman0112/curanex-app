"""WORKFLOW.md A8 for the eye models, natively (ONNX is safe): python tools/evaluate_eye.py (from backend/).

Independent set: sjchoi86 retina_dataset (300 normal, 100 cataract, 101 glaucoma, 100 other retinal disease). Half (seed 42)
settles the unknowns (output polarity, cataract input scale) and the threshold (highest with sensitivity >= 0.90); the
other half is scored once, against normal photos and, harder, against normal + the other diseases.
"""
import hashlib
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from tools.eye_adapter_draft import MODEL_DIR, prep_cataract, prep_glaucoma  # noqa: E402

ROOT = BACKEND / "data" / "eval" / "cataract" / "dataset"
FOLDERS = {"normal": "1_normal", "cataract": "2_cataract", "glaucoma": "2_glaucoma", "retina": "3_retina_disease"}
sessions = {}


def init() -> None:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    for name in ("cataract", "glaucoma"):
        sessions[name] = ort.InferenceSession(str(MODEL_DIR / f"{name}.onnx"), opts, providers=["CPUExecutionProvider"])


def run(name, x):
    s = sessions[name]
    return float(s.run(None, {s.get_inputs()[0].name: x})[0].ravel()[0])


def one(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return [run("cataract", prep_cataract(img, 1.0)), run("cataract", prep_cataract(img, 255.0)), run("glaucoma", prep_glaucoma(img))]


def auc(y, s):
    order = s.argsort()
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
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
    with Pool(6, initializer=init) as pool:
        rows = pool.map(one, [p for p, _ in items], chunksize=8)
    keep = [i for i, r in enumerate(rows) if r is not None]
    S = np.array([rows[i] for i in keep])
    kind = np.array([items[i][1] for i in keep])
    idx = list(range(len(kind)))
    random.Random(42).shuffle(idx)  # noqa: S311 sampling, not security
    va, te = np.array(sorted(idx[: len(idx) // 2])), np.array(sorted(idx[len(idx) // 2 :]))
    meta = json.loads((MODEL_DIR / "convert.json").read_text())
    for name, cols in (("cataract", {1.0: 0, 255.0: 1}), ("glaucoma", {255.0: 2})):
        best = None
        for scale, col in cols.items():
            for inverted in (False, True):
                s = 1 - S[:, col] if inverted else S[:, col]
                m = np.isin(kind[va], [name, "normal"])
                a = auc((kind[va][m] == name).astype(int), s[va][m])
                print(f"{name} scale={scale} inverted={inverted} half-1 AUC {a:.3f}", flush=True)
                if best is None or a > best[0]:
                    best = (a, scale, inverted, s)
        _, scale, inverted, s = best
        m = kind[va] == name
        threshold = float(max(t for t in np.unique(s[va][m]) if (s[va][m] >= t).mean() >= 0.90))
        easy = np.isin(kind[te], [name, "normal"])
        y = (kind[te] == name).astype(int)
        meta[name].update({"scale": scale, "inverted": inverted, "threshold": threshold,
                           "onnx_sha256": hashlib.sha256((MODEL_DIR / f"{name}.onnx").read_bytes()).hexdigest(),
                           "test_vs_normal": report(y[easy], s[te][easy], threshold), "test_vs_everything_else": report(y, s[te], threshold)})
        print(name, json.dumps({k: meta[name][k] for k in ("scale", "inverted", "test_vs_normal", "test_vs_everything_else")}), flush=True)
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
