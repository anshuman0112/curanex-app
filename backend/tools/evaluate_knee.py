"""WORKFLOW.md A8 for the knee model, natively (ONNX is safe): python tools/evaluate_knee.py (from backend/).

Repo's held-out Test-data, one folder per KL grade. Primary metric, fixed before running: KL grade 2 or higher vs 0-1,
using the summed probability of grades 2-4 at 0.5.
"""
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND / "workers" / "tf" / "models" / "ghifar_khder_knee_osteoarthritis_classification"
TEST = BACKEND / "vendor" / "ghifar_khder_knee_osteoarthritis_classification" / "Test-data"
session = None


def init() -> None:
    global session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])


def one(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224))
    return session.run(None, {session.get_inputs()[0].name: np.asarray(img, dtype=np.float32)[np.newaxis] / 255.0})[0][0]


def auc(y, s):
    order = s.argsort()
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        r[s == v] = r[s == v].mean()
    pos, neg = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> None:
    items = [(str(p), g) for g in range(5) for p in sorted((TEST / str(g)).glob("*"))]
    with Pool(6, initializer=init) as pool:
        p = np.array(pool.map(one, [i[0] for i in items], chunksize=8))
    g = np.array([i[1] for i in items])
    y = (g >= 2).astype(int)
    s = p[:, 2:].sum(axis=1)
    f = s >= 0.5
    tp, fn, tn, fp = int((f & (y == 1)).sum()), int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum()), int((f & (y == 0)).sum())
    metrics = {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn), 3),
               "specificity": round(tn / (tn + fp), 3), "auc": round(auc(y, s), 3), "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
               "five_grade_accuracy": round(float((p.argmax(axis=1) == g).mean()), 3), "per_grade_n": np.bincount(g).tolist()}
    (MODEL_DIR / "model_meta.json").write_text(json.dumps({
        "image_size": 224, "onnx_sha256": hashlib.sha256((MODEL_DIR / "model.onnx").read_bytes()).hexdigest(),
        "max_abs_diff": 1.67e-06, "metrics": metrics}, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
