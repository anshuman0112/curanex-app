"""WORKFLOW.md A8 for the caries detector, natively (ONNX is safe): python tools/evaluate_caries.py (from backend/).

DentalAI splits: the repo trained on train + valid in its own copy, so `valid` is not fully independent, but it is used
only to place the threshold (highest with sensitivity >= 0.90). `test` was not used for training and is scored once.
A photo is positive if an annotator marked Caries or Cavity.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from workers.torch.adapters.andreygermanov_yolov8_caries_detector import MODEL_DIR, lesion_score, letterbox  # noqa: E402

DATA = BACKEND / "data" / "eval" / "dentalai"
session = None


def init() -> None:
    global session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])


def one(path: str) -> float:
    x, _, _, _ = letterbox(Image.open(path))
    return lesion_score(session.run(None, {"images": x})[0])[0]


def split(name: str):
    items = []
    for ann in sorted((DATA / name / "ann").glob("*.json")):
        img = DATA / name / "img" / ann.stem
        if not img.exists():
            continue
        titles = {o["classTitle"].lower() for o in json.loads(ann.read_text(encoding="utf-8"))["objects"]}
        items.append((str(img), int(bool(titles & {"caries", "cavity"}))))
    return items


def auc(y, s):
    order = s.argsort()
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    pos, neg = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def report(y, s, t):
    p = s >= t
    tp, fn, tn, fp = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / max(tp + fn, 1), 3),
            "specificity": round(tn / max(tn + fp, 1), 3), "auc": round(auc(y, s), 3), "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def main() -> None:
    with Pool(6, initializer=init) as pool:
        va, te = split("valid"), split("test")
        print(f"valid {len(va)} (positives {sum(y for _, y in va)}), test {len(te)} (positives {sum(y for _, y in te)})", flush=True)
        s_va = np.array(pool.map(one, [p for p, _ in va], chunksize=4))
        y_va = np.array([y for _, y in va])
        threshold = float(max(t for t in np.unique(s_va) if (s_va[y_va == 1] >= t).mean() >= 0.90))
        s_te = np.array(pool.map(one, [p for p, _ in te], chunksize=4))
    y_te = np.array([y for _, y in te])
    metrics = {"threshold": round(threshold, 4), "valid": report(y_va, s_va, threshold), "test": report(y_te, s_te, threshold)}
    print(json.dumps(metrics, indent=1))
    path = MODEL_DIR / "model_meta.json"
    meta = json.loads(path.read_text())
    meta.update({"threshold": threshold, "metrics": metrics})
    path.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
