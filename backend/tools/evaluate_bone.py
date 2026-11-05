"""WORKFLOW.md A8 for joystonmenezes/xray-vision, natively (the repo ships ONNX). Run from backend/: python tools/evaluate_bone.py

MURA valid set from a Kaggle mirror, scored per study as radiologists read it. Half the patients set the threshold (highest
with sensitivity >= 0.85); the other half is scored once. MURA's valid set may have guided the original training, so this is
a weaker test than a new hospital, and the report says so.
"""
import json
import random
import re
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND / "data" / "eval" / "mura"
MODEL = BACKEND / "vendor" / "joystonmenezes_xray_vision" / "model" / "densenet121_mura.onnx"
OUT = BACKEND / "workers" / "torch" / "models" / "joystonmenezes_xray_vision"
session = None


def prep(img: Image.Image) -> np.ndarray:
    """Same steps as the repo: RGB, longest side 256 keeping aspect, black padding, centre crop 224, 0-1."""
    img = img.convert("RGB")
    s = 256 / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
    canvas = Image.new("RGB", (256, 256))
    canvas.paste(img, ((256 - img.width) // 2, (256 - img.height) // 2))
    a = np.asarray(canvas, dtype=np.float32)[16:240, 16:240] / 255.0
    return a[np.newaxis]


def init() -> None:
    global session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])


def one(path: str):
    try:
        x = prep(Image.open(path))
    except OSError:
        return None
    out = session.run(["densenet121/probability:0"], {"inputs:0": x})[0]
    return float(out[0, 1])


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
    zip_path = ROOT / "data.zip"
    if not list(ROOT.rglob("valid/XR_*")):
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if "/valid/" in n and n.lower().endswith((".png", ".jpg"))]
            print("extracting", len(names), flush=True)
            for n in names:
                z.extract(n, ROOT)
    files = [p for p in ROOT.rglob("*.png") if "/valid/" in p.as_posix() and not p.name.startswith(".")]
    print("images", len(files), flush=True)
    with Pool(4, initializer=init) as pool:
        scores = pool.map(one, [str(p) for p in files], chunksize=16)
    studies = defaultdict(list)
    for p, s in zip(files, scores, strict=True):
        m = re.search(r"XR_(\w+)/(patient\d+)/study\d+_(positive|negative)", p.as_posix())
        if s is not None and m:
            studies[m.group(0)].append((s, m.group(1), m.group(2), int(m.group(3) == "positive")))
    rows = [(float(np.mean([v[0] for v in vals])), *vals[0][1:]) for vals in studies.values()]
    patients = sorted({r[2] for r in rows})
    random.Random(42).shuffle(patients)  # noqa: S311 sampling, not security
    tune = set(patients[: len(patients) // 2])
    s = np.array([r[0] for r in rows])
    y = np.array([r[3] for r in rows])
    part = np.array([r[1] for r in rows])
    va = np.array([r[2] in tune for r in rows])
    print("half-1 AUC", round(auc(y[va], s[va]), 3), "inverted", round(auc(y[va], 1 - s[va]), 3), flush=True)
    t = float(max(v for v in np.unique(s[va]) if (s[va][y[va] == 1] >= v).mean() >= 0.85))
    te = ~va
    result = {"threshold": t, "test": report(y[te], s[te], t), "by_part": {k: report(y[te & (part == k)], s[te & (part == k)], t) for k in sorted(set(part))}}
    print(json.dumps(result, indent=0)[:1500], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_meta.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
