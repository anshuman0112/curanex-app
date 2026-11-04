"""Fast feature extraction outside Docker: the model here is ONNX (no pickle), so it's safe to run natively, and Windows
file access is far quicker than through a Docker bind mount. One ONNX thread per worker process.

  python tools/extract_fast.py train vinbig rsna     (from backend/, needs onnxruntime + scikit-image)

Same selections as extract_xrv_features.py / extract_vinbig_features.py. RSNA uses every positive plus as many
negatives (seed 42). Saves after each set in data/eval/, so it can be stopped and resumed.
"""
import csv
import os
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from workers.torch.adapters.mlmed_torchxrayvision import MODEL_DIR, op_norm, preprocess  # noqa: E402

DATA = BACKEND / "data" / "eval"
CLASSES = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
           "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
           "Pulmonary fibrosis"]
session, thresholds = None, None


def init() -> None:
    global session, thresholds
    import json
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    thresholds = np.array(json.loads((MODEL_DIR / "model_meta.json").read_text())["op_threshs"], dtype=np.float32)


def one(path: str):
    try:
        with Image.open(path) as img:
            img.draft("L", (512, 512))
            x = preprocess(img)
    except Exception:  # unreadable file in a public dataset
        return None
    logits, feats = session.run(["logits", "features"], {"image": x})
    return feats[0], op_norm(1 / (1 + np.exp(-logits)), thresholds)[0]


def run(paths: list[str]):
    workers = max(2, (os.cpu_count() or 4) - 2)
    start = time.time()
    with Pool(workers, initializer=init) as pool:
        rows = []
        for i, r in enumerate(pool.imap(one, paths, chunksize=8), 1):
            rows.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(paths)} ({i / (time.time() - start):.1f} images/s)", flush=True)
    return rows


def train_set() -> None:
    items = []
    for folder, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
        for p in sorted((DATA / "chest_xray" / "train" / folder).glob("*.jpeg")):
            m = re.match(r"(person\d+|NORMAL2-IM-\d+|IM-\d+)", p.name)
            items.append((str(p), label, m.group(1) if m else p.stem))
    rows = run([i[0] for i in items])
    keep = [i for i, r in enumerate(rows) if r is not None]
    np.savez(DATA / "features_train.npz", X=np.stack([rows[i][0] for i in keep]), scores=np.stack([rows[i][1] for i in keep]),
             y=np.array([items[i][1] for i in keep]), group=np.array([items[i][2] for i in keep]))
    print(f"train: saved {len(keep)}", flush=True)


def vinbig_set() -> None:
    root = DATA / "vinbig" / "vinbigdata"
    found: dict[str, set] = {}
    with (root / "train.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            found.setdefault(r["image_id"], set()).add(r["class_name"])
    abnormal = sorted(i for i, c in found.items() if c - {"No finding"})
    normal = sorted(i for i, c in found.items() if c == {"No finding"})
    random.Random(42).shuffle(normal)  # noqa: S311 sampling, not security
    ids = [i for i in abnormal + normal[:2500] if (root / "train" / f"{i}.png").exists()]
    rows = run([str(root / "train" / f"{i}.png") for i in ids])
    keep = [i for i, r in enumerate(rows) if r is not None]
    np.savez(DATA / "features_vinbig.npz", X=np.stack([rows[i][0] for i in keep]), scores=np.stack([rows[i][1] for i in keep]),
             labels=np.array([[c in found[ids[i]] for c in CLASSES] for i in keep]), classes=np.array(CLASSES),
             ids=np.array([ids[i] for i in keep]))
    print(f"vinbig: saved {len(keep)} of {len(ids)}", flush=True)


def rsna_set() -> None:
    root = DATA / "rsna"
    with (root / "labels.csv").open(encoding="utf-8") as f:
        rows_csv = list(csv.DictReader(f))
    pos = [r for r in rows_csv if r["target"] == "1"]
    neg = [r for r in rows_csv if r["target"] == "0"]
    random.Random(42).shuffle(neg)  # noqa: S311 sampling, not security
    chosen = pos + neg[:len(pos)]
    rows = run([str(root / "png" / f"{r['patientId']}.png") for r in chosen])
    keep = [i for i, r in enumerate(rows) if r is not None]
    np.savez(DATA / "features_rsna.npz", X=np.stack([rows[i][0] for i in keep]), scores=np.stack([rows[i][1] for i in keep]),
             y=np.array([int(chosen[i]["target"]) for i in keep]))
    print(f"rsna: saved {len(keep)} of {len(chosen)}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        {"train": train_set, "vinbig": vinbig_set, "rsna": rsna_set}[name]()
