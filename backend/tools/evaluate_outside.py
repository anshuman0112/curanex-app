"""Outside tests that decide whether an Unverified model can become Verified. From backend/: python tools/evaluate_outside.py brain|skin

Thresholds stay exactly as they were: nothing is retuned, and each set is scored once.
brain: navoneel brain MRI set, tumor yes/no, leaving out any file identical to a training image.
skin: PAD-UFES-20 smartphone photos from Brazil. Suspicious is melanoma, basal cell, squamous cell or actinic keratosis.
Pass means sensitivity >= 0.80 and AUC >= 0.80.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, roc_auc_score

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from training.train_skin import prep as skin_prep  # noqa: E402

EVAL = BACKEND / "data" / "eval"
IMAGE = (".jpg", ".jpeg", ".png", ".bmp")


def report(y, s, t):
    tn, fp, fn, tp = confusion_matrix(y, s >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, s)), 3)}


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def brain():
    meta = json.loads((BACKEND / "workers/torch/models/multi/brain.json").read_text())
    s = session(BACKEND / "workers/torch/models/multi/brain.onnx")
    known = {hashlib.md5(f.read_bytes()).hexdigest() for f in (EVAL / "brain" / "x").rglob("*") if f.suffix.lower() in IMAGE}  # noqa: S324 finds duplicates
    i = meta["labels"].index("notumor")
    t = meta["thresholds"][i]
    scores, truth, skipped = [], [], 0
    for f in (EVAL / "brain_outside").rglob("*"):
        if f.suffix.lower() not in IMAGE or "brain_tumor_dataset" not in f.as_posix():
            continue
        if hashlib.md5(f.read_bytes()).hexdigest() in known:  # noqa: S324 finds duplicates
            skipped += 1
            continue
        d = f.parent.name.lower()
        if d not in ("yes", "no"):
            continue
        a = skin_prep(f.read_bytes())[np.newaxis]
        scores.append(float(s.run(None, {"image": a})[0][0, i]))
        truth.append(int(d == "yes"))
    y, p = np.array(truth), 1 - np.array(scores)
    r = report(y, p, 1 - t)
    r["overlap_removed"] = skipped
    return r


def skin():
    meta = json.loads((BACKEND / "workers/torch/models/skin_lesion_check/model_meta.json").read_text())
    s = session(BACKEND / "workers/torch/models/skin_lesion_check/model.onnx")
    root = EVAL / "pad"
    table = next(root.rglob("metadata.csv"))
    files = {f.name: f for f in root.rglob("*.png")}
    scores, truth, kinds = [], [], []
    with table.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f = files.get(r["img_id"])
            if f is None:
                continue
            scores.append(float(s.run(None, {"image": skin_prep(f.read_bytes())[np.newaxis]})[0].ravel()[0]))
            truth.append(int(r["diagnostic"] in ("MEL", "BCC", "SCC", "ACK")))
            kinds.append(r["diagnostic"])
    y, p, k = np.array(truth), np.array(scores), np.array(kinds)
    t = meta["threshold"]
    out = report(y, p, t)
    out["found_by_type"] = {d: round(float((p[k == d] >= t).mean()), 3) for d in sorted(set(kinds))}
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    result = brain() if name == "brain" else skin()
    result["passed"] = result["sensitivity"] >= 0.80 and result["auc"] >= 0.80
    print(json.dumps(result), flush=True)
    (EVAL / f"outside_{name}.json").write_text(json.dumps(result, indent=1))
