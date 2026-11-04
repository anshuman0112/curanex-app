"""Cache model features for VinBigData adult X-rays (radiologist labels, not in the model's training data).
Every image with a finding, plus 2,500 "No finding" images (seed 42). Worker image, no network:

  docker run --rm --network none -e PYTHONPATH=/app -v <backend>/data/eval:/data -v <backend>/workers:/app/workers:ro \
      -v <backend>/tools:/tools:ro backend-worker-torch python /tools/extract_vinbig_features.py
"""
import csv
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tools")
from extract_xrv_features import init, one  # noqa: E402

ROOT = Path("/data/vinbig/vinbigdata")
CLASSES = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
           "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
           "Pulmonary fibrosis"]


def main() -> None:
    found: dict[str, set] = {}
    for r in csv.DictReader((ROOT / "train.csv").open(encoding="utf-8")):
        found.setdefault(r["image_id"], set()).add(r["class_name"])
    abnormal = sorted(i for i, c in found.items() if c - {"No finding"})
    normal = sorted(i for i, c in found.items() if c == {"No finding"})
    random.Random(42).shuffle(normal)  # noqa: S311 sampling, not security
    ids = [i for i in abnormal + normal[:2500] if (ROOT / "train" / f"{i}.png").exists()]
    print(f"{len(abnormal)} abnormal, {len(normal)} normal images in labels; using {len(ids)}", flush=True)
    with Pool(4, initializer=init) as pool:
        rows = pool.map(one, [str(ROOT / "train" / f"{i}.png") for i in ids], chunksize=16)
    keep = [i for i, r in enumerate(rows) if r is not None]
    labels = np.array([[c in found[ids[i]] for c in CLASSES] for i in keep])
    np.savez(Path("/data/features_vinbig.npz"), X=np.stack([rows[i][0] for i in keep]), scores=np.stack([rows[i][1] for i in keep]),
             labels=labels, classes=np.array(CLASSES), ids=np.array([ids[i] for i in keep]))
    print(f"saved {len(keep)} of {len(ids)}", flush=True)


if __name__ == "__main__":
    main()
