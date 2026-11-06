"""Outside test for the malaria check: BBBC041 (P. vivax, different lab, stain and cameras). From backend/: python tools/evaluate_malaria_outside.py

Cells are cropped from whole slides using the published boxes. The threshold stays as trained; scored once. Infected means ring,
trophozoite, schizont or gametocyte; leukocytes and 'difficult' boxes are left out. Pass is sensitivity and AUC >= 0.80.
"""
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND / "data" / "eval" / "bbbc041"
MODEL = BACKEND / "workers" / "torch" / "models" / "multi"
INFECTED = {"ring", "trophozoite", "schizont", "gametocyte"}
PER_CLASS = 800


def prep(img, mean, std):
    img = img.resize((224, 224), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255
    return ((a - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)).transpose(2, 0, 1)[np.newaxis]


def main() -> None:
    meta = json.loads((MODEL / "malaria.json").read_text())
    o = ort.SessionOptions()
    o.intra_op_num_threads = 2
    s = ort.InferenceSession(str(MODEL / "malaria.onnx"), o, providers=["CPUExecutionProvider"])
    z = zipfile.ZipFile(ROOT / "data.zip")
    names = z.namelist()
    tables = [n for n in names if n.endswith(("training.json", "test.json"))]
    files = {Path(n).name: n for n in names if n.lower().endswith(".png")}
    boxes = {0: [], 1: []}
    for t in tables:
        for entry in json.loads(z.read(t)):
            name = Path(entry["image"]["pathname"]).name
            for b in entry["objects"]:
                c = b["category"]
                if c == "red blood cell":
                    boxes[0].append((name, b["bounding_box"]))
                elif c in INFECTED:
                    boxes[1].append((name, b["bounding_box"]))
    rng = random.Random(42)  # noqa: S311 sampling, not security
    print({k: len(v) for k, v in boxes.items()}, flush=True)
    pick = [(n, b, y) for y in (0, 1) for n, b in rng.sample(boxes[y], min(PER_CLASS, len(boxes[y])))]
    pick.sort(key=lambda r: r[0])
    clear = meta["labels"].index("Uninfected")
    t = meta["thresholds"][clear]
    score, truth, cache_name, page = [], [], None, None
    for name, b, y in pick:
        if name not in files:
            continue
        if name != cache_name:
            page, cache_name = Image.open(io.BytesIO(z.read(files[name]))).convert("RGB"), name
        lo, hi = b["minimum"], b["maximum"]
        crop = page.crop((lo["c"], lo["r"], hi["c"], hi["r"]))
        p = s.run(None, {"image": prep(crop, meta["mean"], meta["std"])})[0][0]
        score.append(1 - float(p[clear]))
        truth.append(y)
    y, p = np.array(truth), np.array(score)
    tn, fp, fn, tp = confusion_matrix(y, p >= 1 - t, labels=[0, 1]).ravel()
    out = {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
           "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, p)), 3)}
    out["passed"] = out["sensitivity"] >= 0.80 and out["auc"] >= 0.80
    print(json.dumps(out), flush=True)
    (BACKEND / "data" / "eval" / "outside_malaria.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
