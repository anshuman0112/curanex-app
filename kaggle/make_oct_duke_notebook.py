"""Kaggle CPU notebook: AMD-only OCT check, confirmation test on Duke (Srinivasan 2014), model unchanged.
Run from the repo root: python kaggle/make_oct_duke_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Curanex: OCT check for AMD, confirmation on Duke / Srinivasan 2014 (CPU)

The fine-tuned OCT model (Kermany 2018) passed AMD on OCTDL (sensitivity 0.904, AUC 0.993) but failed DME, so the combined check
was not Verified. **Rules fixed before running (2026-11-11):** an AMD-only check: AMD score = P(CNV) + P(drusen), threshold
min(CNV, drusen thresholds) = 0.822, both unchanged from the OCTDL run. DME is not reported.
**Confirmation test, scored once:** Duke / Srinivasan 2014 (USA): 15 AMD and 15 normal eyes (volumes). One score per eye =
90th percentile of its B-scan AMD scores. Verified only if eye-level sensitivity and AUC are both >= 0.80 (and OCTDL already passed).
B-scan level numbers are reported too, for the record."""

CODE = [r'''!pip install -q onnxruntime''',
r'''import glob, json, os, re
import numpy as np, onnxruntime as ort
from PIL import Image
from sklearn.metrics import roc_auc_score
onx = glob.glob("/kaggle/input/**/oct_ft.onnx", recursive=True)[0]
meta = json.load(open(glob.glob("/kaggle/input/**/oct_ft.json", recursive=True)[0]))
sess = ort.InferenceSession(onx)
MEAN, STD = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)
T_AMD = min(meta["thresholds"][0], meta["thresholds"][2])
print("AMD threshold", T_AMD)


def prep(f):  # Resize(256) + CenterCrop(224) + normalise, as in training and in the app
    im = Image.open(f).convert("RGB"); w, h = im.size; s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    a = (np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1)


files = [f for f in glob.glob("/kaggle/input/**/*.*", recursive=True) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))
         and os.path.basename(os.path.dirname(f)).upper() in ("AMD", "NORMAL")]
print("B-scans:", {k: sum(1 for f in files if os.path.basename(os.path.dirname(f)).upper() == k) for k in ("AMD", "NORMAL")})
scores = {}
for i in range(0, len(files), 64):
    batch = files[i:i + 64]
    p = sess.run(None, {sess.get_inputs()[0].name: np.stack([prep(f) for f in batch])})[0]
    if p.max() > 1 or p.min() < 0:
        p = np.exp(p) / np.exp(p).sum(1, keepdims=True)
    for f, row in zip(batch, p):
        scores[f] = float(row[0] + row[2])


def report(y, s, t):
    y, s = np.array(y), np.array(s); fl = s >= t
    r = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(fl[y == 1].mean()), 3),
         "specificity": round(float((~fl[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, s)), 3)}
    low = min(r["sensitivity"], r["auc"]); r["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
    return r


eyes = {}
for f, s in scores.items():
    eyes.setdefault(re.split(r"[_ ]", os.path.basename(f))[0], []).append(s)
ey = [1 if k.upper().startswith("AMD") else 0 for k in eyes]
es = [float(np.percentile(v, 90)) for v in eyes.values()]
res = {"eye_level": report(ey, es, T_AMD),
       "bscan_level": report([1 if os.path.basename(os.path.dirname(f)).upper() == "AMD" else 0 for f in scores], list(scores.values()), T_AMD)}
print("CONFIRMATION TEST on Duke (once):", res)
json.dump(res, open("/kaggle/working/oct_duke.json", "w"), indent=1)''']

cells = [{"cell_type": "markdown", "metadata": {}, "source": INTRO}]
cells += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in CODE]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_oct_duke"
d.mkdir(exist_ok=True)
(d / "oct_duke.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-oct-duke", "title": "Curanex oct duke", "code_file": "oct_duke.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": ["buffyhridoy/oct-dataset-duke-srinivasan-2014"], "competition_sources": [], "kernel_sources": ["anshu004/curanex-oct-ft"]}, indent=1))
print("written", d)
