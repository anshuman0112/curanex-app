"""Outside tests for models from train_folder_head.py. From backend/: python tools/evaluate_outside_multi.py lung|kidney|dr

Thresholds stay as trained and each set is scored once. Any file identical to a training image is removed first.
lung: separate chest CT collection, cancer against normal. kidney: separate kidney CT collection, tumor against the rest.
dr: IDRiD (India), any retinopathy against none. Pass means sensitivity >= 0.80 and AUC >= 0.80.
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

BACKEND = Path(__file__).resolve().parents[1]
EVAL = BACKEND / "data" / "eval"
MODELS = BACKEND / "workers" / "torch" / "models" / "multi"
IMAGE = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
PER_CLASS = 300


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 finds duplicates, not security


def prep(path, meta):
    img = Image.open(path).convert("RGB")
    s = 256 / min(img.size)
    img = img.resize((max(224, round(img.width * s)), max(224, round(img.height * s))), Image.BILINEAR)
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    a = np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255
    mean, std = np.array(meta["mean"], dtype=np.float32), np.array(meta["std"], dtype=np.float32)
    return ((a - mean) / std).transpose(2, 0, 1)[np.newaxis]


def known(folder: Path) -> set:
    return {md5(f) for f in folder.rglob("*") if f.suffix.lower() in IMAGE}


def items(name):
    rng = random.Random(42)  # noqa: S311 sampling, not security
    if name == "lung":
        root = EVAL / "lung_outside" / "x" / "Data"
        files = [f for f in root.rglob("*") if f.suffix.lower() in IMAGE]
        return [(f, int("normal" not in f.parent.name.lower())) for f in files], "lungct", "Normal", EVAL / "lungct" / "clean"
    if name == "kidney":
        out = []
        for c in ("cyst", "normal", "stone", "tumor"):
            fs = sorted((EVAL / "kidney_outside" / "x" / "dataset" / c).glob("*"))
            out += [(f, int(c == "tumor")) for f in rng.sample(fs, min(PER_CLASS, len(fs)))]
        return out, "kidneyct", "Tumor", EVAL / "kidneyct" / "x"
    root = EVAL / "idrid" / "x"
    with (root / "idrid_labels.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    print(rows[:2], flush=True)
    head = [h.lower() for h in rows[0]]
    i_name = next(i for i, h in enumerate(head) if "id" in h or "image" in h)
    i_grade = next(i for i, h in enumerate(head) if "diag" in h or "grade" in h or "retin" in h or "dr" in h and i != i_name)
    files = {f.stem: f for f in (root / "Imagenes").rglob("*") if f.suffix.lower() in IMAGE}
    return [(files[Path(r[i_name]).stem], int(float(r[i_grade]) > 0)) for r in rows[1:] if Path(r[i_name]).stem in files], "dr", "No_DR", EVAL / "dr" / "colored_images"


def main(name: str) -> None:
    rows, model, target, train = items(name)
    meta = json.loads((MODELS / f"{model}.json").read_text())
    seen = known(train)
    keep = [(f, y) for f, y in rows if md5(f) not in seen]
    print(name, "images", len(rows), "after removing training copies", len(keep), "positives", sum(y for _, y in keep), flush=True)
    o = ort.SessionOptions()
    o.intra_op_num_threads = 3
    s = ort.InferenceSession(str(MODELS / f"{model}.onnx"), o, providers=["CPUExecutionProvider"])
    i = meta["labels"].index(target)
    t = meta["thresholds"][i]
    score, truth = [], []
    for f, y in keep:
        try:
            p = s.run(None, {"image": prep(f, meta)})[0][0]
        except OSError:
            continue
        score.append(float(p[i]))
        truth.append(y)
    y, p = np.array(truth), np.array(score)
    if target in ("Normal", "No_DR"):  # the clear class: a low score means disease
        p, t = 1 - p, 1 - t
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    r = {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
         "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, p)), 3), "copies_removed": len(rows) - len(keep)}
    r["passed"] = r["sensitivity"] >= 0.80 and r["auc"] >= 0.80
    print(json.dumps(r), flush=True)
    (EVAL / f"outside_{name}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
