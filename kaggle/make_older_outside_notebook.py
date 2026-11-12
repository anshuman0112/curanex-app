"""Kaggle CPU notebook (2026-11-12): outside tests for two older Verified image models, run with the app's own adapters.
Run from the repo root: python kaggle/make_older_outside_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTRO = """# Curanex: outside tests for the knee X-ray and chest X-ray checks (CPU)

Both were marked Verified earlier on held-out data from their own source. **Rules fixed before running (2026-11-12):** the app's own
adapter code and model files, unchanged; each image scored once.
- **Knee X-ray (KL grade 2+ vs 0-1):** Digital Knee X-ray Images (Gornale et al., India), grades by Medical Expert I (Expert II reported
  too). Flagged = the app's label "Possible moderate arthritis signs"; AUC on the "Moderate or worse" score.
- **Chest X-ray (adult abnormal X-ray):** NLM Shenzhen (China) and Montgomery County (USA) sets, which are not in torchxrayvision's
  training data; abnormal = TB (file name ends _1). Flagged = any app label other than "No clear abnormality found"; AUC on the
  "Abnormal X-ray" score.
- Each stays Verified only if sensitivity and AUC are both >= 0.80."""

CODE = r'''import base64, glob, io, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
import shutil
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/common/adapter.py", recursive=True)[0]))
shutil.copytree(src, "/kaggle/working/app/workers", dirs_exist_ok=True)
sys.path.insert(0, "/kaggle/working/app")
from workers.tf.adapters.ghifar_khder_knee_osteoarthritis_classification import Adapter as Knee
from workers.torch.adapters.mlmed_torchxrayvision import Adapter as Cxr


def report(y, s, f):
    y, s, f = map(np.array, (y, s, f))
    r = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(f[y == 1].mean()), 3),
         "specificity": round(float((~f[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, s)), 3)}
    low = min(r["sensitivity"], r["auc"]); r["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
    return r


b64 = lambda f: base64.b64encode(open(f, "rb").read()).decode()
out = {}
knee = Knee(); knee.load()
rows = {}
for f in glob.glob("/kaggle/input/**/*", recursive=True):
    if not f.lower().endswith((".png", ".jpg", ".jpeg")) or "adapters" in f:
        continue
    folder = os.path.basename(os.path.dirname(f)); m = re.match(r"(\d)", folder)
    expert = "II" if re.search(r"Expert[-_ ]?II", f, re.I) else "I" if re.search(r"Expert[-_ ]?I", f, re.I) else None
    if m and expert:
        rows.setdefault(expert, []).append((f, int(int(m.group(1)) >= 2)))
print("knee images:", {k: (len(v), sum(y for _, y in v)) for k, v in rows.items()})
assert rows.get("I")
for expert in ("I", "II"):
    ys, ss, fs, refused = [], [], [], 0
    for f, y in rows.get(expert, []):
        try:
            p = knee.predict({"image_b64": b64(f)})
        except ValueError:
            refused += 1; continue
        ys.append(y); ss.append(p.probabilities["Moderate or worse"]); fs.append(p.label == "Possible moderate arthritis signs")
    if ys:
        out[f"knee_expert_{expert}"] = report(ys, ss, fs) | {"refused": refused}
        print(f"OUTSIDE TEST knee (Expert {expert}, once):", out[f"knee_expert_{expert}"], flush=True)
cxr = Cxr(); cxr.load()
test = sorted({(f, int(os.path.basename(f).endswith("_1.png"))) for f in glob.glob("/kaggle/input/**/*.png", recursive=True)
               if os.path.basename(f).startswith(("CHNCXR", "MCUCXR")) and "mask" not in f.lower()})
print("NLM chest X-rays:", len(test), "TB:", sum(y for _, y in test))
ys, ss, fs, refused = [], [], [], 0
for f, y in test:
    try:
        p = cxr.predict({"image_b64": b64(f)})
    except ValueError:
        refused += 1; continue
    ys.append(y); ss.append(p.probabilities["Abnormal X-ray"]); fs.append(p.label != "No clear abnormality found")
out["chest_xray_nlm"] = report(ys, ss, fs) | {"refused": refused}
print("OUTSIDE TEST chest X-ray (NLM, once):", out["chest_xray_nlm"])
json.dump(out, open("/kaggle/working/older_outside.json", "w"), indent=1)'''

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": INTRO},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "!pip install -q onnxruntime scikit-image"},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_older_outside"
d.mkdir(exist_ok=True)
(d / "older_outside.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-older-outside", "title": "Curanex older outside", "code_file": "older_outside.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": ["anshu004/curanex-app-adapters", "orvile/digital-knee-x-ray-images", "kmader/pulmonary-chest-xray-abnormalities"],
    "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", d)
