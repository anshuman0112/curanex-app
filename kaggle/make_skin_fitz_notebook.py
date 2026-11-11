"""Kaggle CPU notebook: outside test of the skin-photo check (DermNet model) on Fitzpatrick17k, model unchanged.
Run from the repo root: python kaggle/make_skin_fitz_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Curanex: skin-photo check, outside test on Fitzpatrick17k (CPU)

The skin check was trained on DermNet NZ photos. Fitzpatrick17k photos come from other atlases (DermaAmin, Atlas Dermatologico).
**Rules fixed before running (2026-11-11):**
- Model, preprocessing and per-group thresholds unchanged. Only the groups the app shows ("passed" in the model file) are scored.
- Fitzpatrick labels are mapped to DermNet groups with the fixed list below (labels that fit no group, or fit loosely, are left out).
- For each group with at least 30 mapped photos: AUC = that group's score, group photos vs all other mapped photos; sensitivity =
  share of the group's photos where the group is in the app's top-3 shortlist (what the app shows). Top-1 accuracy is reported too.
- **Scored once.** Verified if the averages across scored groups are both >= 0.80 (the same rule as the first test); each group's
  numbers are reported, and groups below 0.80 on either would be dropped from the app."""

MAP = {
    "Acne and Rosacea Photos": ["acne", "rosacea", "rhinophyma"],
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": ["actinic keratos", "basal cell carcinoma", "squamous cell carcinoma"],
    "Bullous Disease Photos": ["pemphigus", "pemphigoid", "epidermolysis bullosa", "hailey hailey", "dermatitis herpetiformis"],
    "Eczema Photos": ["eczema"],
    "Exanthems and Drug Eruptions": ["drug eruption", "fixed eruption", "drug induced pigmentary"],
    "Hair Loss Photos Alopecia and other Hair Diseases": ["alopecia", "male pattern baldness"],
    "Light Diseases and Disorders of Pigmentation": ["vitiligo", "photodermatoses", "porphyria", "melasma", "solar lentigo"],
    "Lupus and other Connective Tissue diseases": ["lupus", "scleroderma", "dermatomyositis", "morphea"],
    "Melanoma Skin Cancer Nevi and Moles": ["melanoma", "lentigo maligna", "nevus", "naevus"],
    "Poison Ivy Photos and other Contact Dermatitis": ["contact dermatitis"],
    "Psoriasis pictures Lichen Planus and related diseases": ["psoriasis", "lichen planus", "pityriasis rosea", "pityriasis rubra pilaris"],
    "Scabies Lyme Disease and other Infestations and Bites": ["scabies", "lyme", "tungiasis"],
    "Seborrheic Keratoses and other Benign Tumors": ["seborrheic keratosis", "dermatofibroma", "neurofibromatosis", "syringoma", "xanthoma"],
    "Systemic Disease": ["necrobiosis lipoidica", "acanthosis nigricans", "sarcoidosis"],
    "Urticaria Hives": ["urticaria"],
    "Vascular Tumors": ["pyogenic granuloma", "port wine stain", "hemangioma", "haemangioma", "kaposi"],
    "Vasculitis Photos": ["vasculitis"],
    "Warts Molluscum and other Viral Infections": ["wart", "molluscum"],
    "Tinea Ringworm Candidiasis and other Fungal Infections": ["tinea", "candid"],
}
EXCLUDE = ["urticaria pigmentosa", "becker nevus", "nevus sebaceous", "halo nevus"]  # different disease despite the word, or not a mole

CODE = [r'''!pip install -q onnxruntime''',
r'''import glob, json, os, re
import numpy as np, pandas as pd, onnxruntime as ort
from PIL import Image
from sklearn.metrics import roc_auc_score
MAP = ''' + repr(MAP) + '''
EXCLUDE = ''' + repr(EXCLUDE) + r'''
meta = json.load(open(glob.glob("/kaggle/input/**/dermnet.json", recursive=True)[0]))
sess = ort.InferenceSession(glob.glob("/kaggle/input/**/dermnet.onnx", recursive=True)[0])
LAB = meta["labels"]; SHOWN = [i for i, l in enumerate(LAB) if l in meta["passed"]]
MEAN, STD = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", s.lower())).strip()


def group_of(label):
    l = norm(label)
    if any(e in l for e in EXCLUDE):
        return None
    hits = [g for g, pats in MAP.items() if any(p in l for p in pats)]
    return hits[0] if len(hits) == 1 else None


imgs = [f for f in glob.glob("/kaggle/input/**/*", recursive=True) if f.lower().endswith((".jpg", ".jpeg", ".png")) and "dermnet" not in f.lower()]
by_md5 = {os.path.splitext(os.path.basename(f))[0]: f for f in imgs}
rows = []
for c in glob.glob("/kaggle/input/**/*.csv", recursive=True):
    d = pd.read_csv(c)
    if {"md5hash", "label"} <= set(d.columns):
        rows += [(by_md5[h], l) for h, l in zip(d.md5hash, d.label) if h in by_md5]
if not rows:  # file names carry the label, e.g. acne_vulgaris271.jpg
    rows = [(f, re.sub(r"[\d_]+$", "", os.path.splitext(os.path.basename(f))[0])) for f in imgs]
rows = sorted(set(rows))
print("photos found:", len(imgs), "| labelled:", len(rows), "| labels:", len({l for _, l in rows}))
items = [(f, group_of(l)) for f, l in rows]
items = [(f, g) for f, g in items if g in meta["passed"]]
counts = pd.Series([g for _, g in items]).value_counts()
print(counts.to_string())
assert len(items) > 500, "too few mapped photos - check the layout before scoring"''',
r'''def prep(f):
    im = Image.open(f).convert("RGB"); w, h = im.size; s = 256 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return ((np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


P, keep = [], []
for i in range(0, len(items), 64):
    batch = []
    for f, g in items[i:i + 64]:
        try:
            batch.append(prep(f)); keep.append(g)
        except Exception:
            pass
    if batch:
        P.append(sess.run(None, {"image": np.stack(batch)})[0])
P = np.concatenate(P); G = np.array(keep)
top3 = np.array([[i for i in sorted(SHOWN, key=lambda i: -p[i])[:3]] for p in P])
top1 = top3[:, 0]
res = {}
for g in counts.index:
    if counts[g] < 30:
        continue
    gi = LAB.index(g); y = (G == g).astype(int)
    res[g] = {"n": int(y.sum()), "sensitivity_top3": round(float((top3[y == 1] == gi).any(axis=1).mean()), 3),
              "auc": round(float(roc_auc_score(y, P[:, gi])), 3)}
avg = {"groups": len(res), "sensitivity": round(float(np.mean([r["sensitivity_top3"] for r in res.values()])), 3),
       "auc": round(float(np.mean([r["auc"] for r in res.values()])), 3),
       "top1_accuracy": round(float(np.mean([LAB[t] == g for t, g in zip(top1, G)])), 3), "photos": int(len(G))}
low = min(avg["sensitivity"], avg["auc"]); avg["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
print("OUTSIDE TEST on Fitzpatrick17k (once):", avg)
for g, r in res.items():
    print("  ", g, r)
json.dump({"average": avg, "groups": res}, open("/kaggle/working/skin_fitz.json", "w"), indent=1)''']

cells = [{"cell_type": "markdown", "metadata": {}, "source": INTRO}]
cells += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in CODE]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_skin_fitz"
d.mkdir(exist_ok=True)
(d / "skin_fitz.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-skin-fitz", "title": "Curanex skin fitz", "code_file": "skin_fitz.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": ["mobaswiralfarabi/fitzpatrick17k-original", "anshu004/curanex-skin-model"], "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", d)
