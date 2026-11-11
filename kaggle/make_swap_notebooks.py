"""Builds two Kaggle GPU notebooks (2026-11-11). Run from the repo root: python kaggle/make_swap_notebooks.py

- run_endoscopy/: train on HyperKvasir, test once on Kvasir v2 (endoscopy check + oesophagitis check).
- run_dme/: train on Messidor-2, test once on IDRiD (macular oedema check).
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

COMMON = '''import glob, hashlib, json, os, random
import numpy as np, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

torch.manual_seed(42); random.seed(42); np.random.seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8)), dtype=np.float32)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).ravel()), 2)


def near(h, bank):
    return bool((np.unpackbits(np.bitwise_xor(bank, np.uint64(h)).view(np.uint8).reshape(-1, 8), axis=1).sum(1) <= 6).any())


train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomRotation(20), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.05), T.RandomApply([T.GaussianBlur(5)], p=0.3),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])  # what the app does


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(Image.open(self.items[i][0]).convert("RGB")), self.items[i][1]


def loader(items, tf, shuffle=False):
    return torch.utils.data.DataLoader(DS(items, tf), batch_size=32, shuffle=shuffle, num_workers=2)


def train_model(rows, n_classes, epochs):
    y = np.array([r[1] for r in rows])
    tr, va = train_test_split(np.arange(len(rows)), test_size=0.15, random_state=42, stratify=y)
    net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)
    net = net.to(DEV)
    counts = np.bincount(y[tr], minlength=n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(tr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=DEV))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl_tr, dl_va = loader([rows[i] for i in tr], train_tf, True), loader([rows[i] for i in va], eval_tf)

    def predict(dl):
        net.eval()
        out = []
        with torch.no_grad():
            for x, _ in dl:
                out.append(torch.softmax(net(x.to(DEV)), 1).cpu().numpy())
        return np.concatenate(out)

    best, state = -1, None
    for ep in range(epochs):
        net.train()
        for x, lab in dl_tr:
            opt.zero_grad()
            loss = loss_fn(net(x.to(DEV)), lab.to(DEV))
            loss.backward()
            opt.step()
        sched.step()
        pv = predict(dl_va)
        auc = roc_auc_score(y[va], pv[:, 1] if n_classes == 2 else pv, **({} if n_classes == 2 else {"multi_class": "ovr", "average": "macro"}))
        print(f"epoch {ep + 1}: validation AUC {auc:.4f}")
        if auc > best:
            best, state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    pv = predict(dl_va)
    thresholds = []
    for i in range(n_classes):
        pos = pv[y[va] == i, i]
        thresholds.append(float(max(v for v in np.unique(pos) if (pos >= v).mean() >= 0.85)) if len(pos) else 0.5)
    return net, predict, thresholds, best


def report(y, s, t):
    f = s >= t
    r = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(f[y == 1].mean()), 3),
         "specificity": round(float((~f[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, s)), 3)}
    low = min(r["sensitivity"], r["auc"])
    r["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
    return r


def export(net, name, meta):
    class Full(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return torch.softmax(self.n(x), 1)

    full = Full(net.cpu()).eval()
    path = f"/kaggle/working/{name}.onnx"
    args = dict(input_names=["image"], output_names=["probs"], opset_version=17, dynamic_axes={"image": {0: "n"}})
    try:
        torch.onnx.export(full, torch.randn(1, 3, 224, 224), path, dynamo=False, **args)
    except TypeError:
        torch.onnx.export(full, torch.randn(1, 3, 224, 224), path, **args)
    meta |= {"mean": MEAN, "std": STD, "onnx_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(meta, open(f"/kaggle/working/{name}.json", "w"), indent=1)
    print("wrote", name)'''


def notebook(folder, intro, cells_code, meta):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": COMMON}]
    cells += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in cells_code]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    d = HERE / folder
    d.mkdir(exist_ok=True)
    (d / meta["code_file"]).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({"language": "python", "kernel_type": "notebook", "is_private": True,
                                                         "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
                                                         "competition_sources": [], "kernel_sources": [], **meta}, indent=1))
    print("written", d)


ENDO_INTRO = """# Curanex: endoscopy models, swap design (GPU)

**Rules fixed before running (2026-11-11):**
- Train an 8-class model on HyperKvasir (Baerum Hospital) with the Kvasir class names; 15% stratified validation; best epoch by macro AUC;
  per-class threshold = highest with validation sensitivity >= 0.85.
- **Outside test, scored once:** Kvasir v2 (Vestre Viken). Test images within 6 bits (difference hash) of any training image are removed.
- Findings: colon polyps vs normal cecum; ulcerative colitis vs normal cecum; oesophagitis vs normal Z-line.
- Endoscopy check Verified only if polyps and ulcerative colitis both reach sensitivity and AUC >= 0.80; oesophagitis check Verified at >= 0.80."""

ENDO_DATA = '''LABELS = ["dyed-lifted-polyps", "dyed-resection-margins", "esophagitis", "normal-cecum", "normal-pylorus", "normal-z-line", "polyps",
          "ulcerative-colitis"]
HK_MAP = {"dyed-lifted-polyps": "dyed-lifted-polyps", "dyed-resection-margins": "dyed-resection-margins", "esophagitis-a": "esophagitis",
          "esophagitis-b-d": "esophagitis", "cecum": "normal-cecum", "pylorus": "normal-pylorus", "z-line": "normal-z-line", "polyps": "polyps"}
rows = []
for f in glob.glob("/kaggle/input/**/labeled-images/**/*.jpg", recursive=True):
    folder = os.path.basename(os.path.dirname(f))
    lab = "ulcerative-colitis" if folder.startswith("ulcerative-colitis") else HK_MAP.get(folder)
    if lab:
        rows.append((f, LABELS.index(lab)))
print("HyperKvasir training images:", {l: sum(1 for _, y in rows if y == i) for i, l in enumerate(LABELS)})
bank = np.array([dhash(f) for f, _ in rows], dtype=np.uint64)
test, dropped = [], 0
for i, lab in enumerate(LABELS):
    files = [f for f in glob.glob(f"/kaggle/input/**/{lab}/**/*.jpg", recursive=True) if "labeled-images" not in f]
    for f in sorted(set(files)):
        if near(dhash(f), bank):
            dropped += 1
            continue
        test.append((f, i))
print("Kvasir v2 test images kept:", len(test), "copies of training removed:", dropped)'''

ENDO_RUN = '''net, predict, thr, best = train_model(rows, len(LABELS), epochs=12)
pt = predict(loader(test, eval_tf))
yt = np.array([y for _, y in test])
results = {}
for name, pos, neg in (("colon polyps", "polyps", "normal-cecum"), ("ulcerative colitis", "ulcerative-colitis", "normal-cecum"),
                       ("oesophagitis", "esophagitis", "normal-z-line")):
    i, j = LABELS.index(pos), LABELS.index(neg)
    keep = (yt == i) | (yt == j)
    results[name] = report((yt[keep] == i).astype(int), pt[keep, i], thr[i])
    print("OUTSIDE TEST on Kvasir v2 (once):", name, results[name])
export(net, "endoscopy_hk", {"id": "endoscopy_hk", "title": "Endoscopy (HyperKvasir-trained)", "labels": LABELS, "thresholds": thr,
                             "validation_auc": round(best, 4), "outside_test": results, "images": len(rows),
                             "trained_on": f"{len(rows)} HyperKvasir images (Baerum Hospital, Norway), full fine-tuning on a Kaggle GPU",
                             "outside_set": "Kvasir v2 (Vestre Viken, Norway), copies of training images removed"})'''

DME_INTRO = """# Curanex: macular oedema, swap design (GPU)

**Rules fixed before running (2026-11-11):**
- Train on Messidor-2 (France) with Google's adjudicated DME grades (gradable images only); 15% stratified validation; best epoch by AUC;
  threshold = highest with validation sensitivity >= 0.85.
- **Outside test, scored once:** IDRiD (India), all images with a DME grade (1 or 2 = oedema). Copies of training images removed.
- Verified if outside sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

DME_DATA = '''import pandas as pd
LABELS = ["Macular_edema", "No_macular_edema"]  # same labels as the live model
grades = pd.read_csv(glob.glob("/kaggle/input/**/messidor_data.csv", recursive=True)[0])
grades = grades[grades.columns[:4]]
grades.columns = ["image_id", "dr", "dme", "gradable"]
grades = grades.dropna(subset=["dme"])
grades = grades[grades["gradable"] == 1]
files = {os.path.basename(f): f for f in glob.glob("/kaggle/input/**/preprocess/*.png", recursive=True)}
rows = [(files[r.image_id], 0 if r.dme == 1 else 1) for r in grades.itertuples() if r.image_id in files]
print("Messidor-2 training images:", len(rows), "with oedema:", sum(1 for _, y in rows if y == 0))
bank = np.array([dhash(f) for f, _ in rows], dtype=np.uint64)
lab = pd.read_csv(glob.glob("/kaggle/input/**/idrid_labels.csv", recursive=True)[0])
lab = lab[lab.columns[:3]]
lab.columns = ["id", "dr", "dme"]
idr = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob("/kaggle/input/**/Imagenes/**/*.jpg", recursive=True)}
test = [(idr[r.id], int(r.dme in (1, 2))) for r in lab.itertuples() if r.id in idr and str(r.dme) != "nan"]
test = [(f, y) for f, y in test if not near(dhash(f), bank)]
print("IDRiD test images:", len(test), "with oedema:", sum(y for _, y in test))'''

DME_RUN = '''net, predict, thr, best = train_model(rows, 2, epochs=15)
pt = predict(loader(test, eval_tf))[:, 0]
yt = np.array([y for _, y in test])
res = report(yt, pt, thr[0])
print("OUTSIDE TEST on IDRiD (once):", res)
export(net, "dme_messidor", {"id": "dme_messidor", "title": "Macular oedema (Messidor-trained)", "labels": LABELS, "thresholds": thr,
                             "validation_auc": round(best, 4), "outside_test": res, "images": len(rows),
                             "trained_on": f"{len(rows)} Messidor-2 retina photos (France) with adjudicated DME grades, full fine-tuning on a Kaggle GPU",
                             "outside_set": "IDRiD (India), all images with a DME grade"})'''

notebook("run_endoscopy", ENDO_INTRO, [ENDO_DATA, ENDO_RUN],
         {"id": "anshu004/curanex-endoscopy-swap", "title": "Curanex endoscopy swap", "code_file": "endoscopy_swap.ipynb",
          "dataset_sources": ["faustkkk/hyper-kvasir-labeled-images", "plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset"]})
notebook("run_dme", DME_INTRO, [DME_DATA, DME_RUN],
         {"id": "anshu004/curanex-dme-swap", "title": "Curanex dme swap", "code_file": "dme_swap.ipynb",
          "dataset_sources": ["google-brain/messidor2-dr-grades", "mariaherrerot/messidor2preprocess", "mariaherrerot/idrid-dataset"]})

KID_INTRO = """# Curanex: kidney stone CT, swap design (GPU)

The live kidney CT check (trained on the CT-KIDNEY Normal-Cyst-Tumor-Stone set) scored AUC 0.52 on the axial kidney stone collection.
**Rules fixed before running (2026-11-11):**
- Train on the axial kidney stone collection, Original Dataset only (Stone vs Non-Stone); 15% stratified validation; threshold for sensitivity >= 0.85.
- **Outside test, scored once:** CT-KIDNEY Normal-Cyst-Tumor-Stone, Stone vs Normal; images within 6 bits of any training image removed.
- Verified if outside sensitivity and AUC >= 0.80; live at >= 0.75; otherwise off."""

KID_DATA = '''LABELS = ["Stone", "Normal"]
rows = []
for f in glob.glob("/kaggle/input/**/Original Dataset/**/*", recursive=True):
    if f.lower().endswith((".jpg", ".png")):
        folder = os.path.basename(os.path.dirname(f)).lower()
        rows.append((f, 0 if folder == "stone" else 1))
print("training (axial collection):", len(rows), "stone:", sum(1 for _, y in rows if y == 0))
bank = np.array([dhash(f) for f, _ in rows], dtype=np.uint64)
test, dropped = [], 0
for f in glob.glob("/kaggle/input/**/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/**/*", recursive=True):
    folder = os.path.basename(os.path.dirname(f)).lower()
    if not f.lower().endswith((".jpg", ".png")) or folder not in ("stone", "normal"):
        continue
    if near(dhash(f), bank):
        dropped += 1
        continue
    test.append((f, 1 if folder == "stone" else 0))
print("CT-KIDNEY test images:", len(test), "stone:", sum(y for _, y in test), "copies removed:", dropped)'''

KID_RUN = '''net, predict, thr, best = train_model(rows, 2, epochs=12)
pt = predict(loader(test, eval_tf))[:, 0]
yt = np.array([y for _, y in test])
res = report(yt, pt, thr[0])
print("OUTSIDE TEST on CT-KIDNEY (once):", res)
export(net, "kidney_stone_axial", {"id": "kidney_stone_axial", "title": "Kidney stone CT (axial-trained)", "labels": LABELS, "thresholds": thr,
                                   "validation_auc": round(best, 4), "outside_test": res, "images": len(rows),
                                   "trained_on": f"{len(rows)} axial kidney CT images (stone vs no stone), full fine-tuning on a Kaggle GPU",
                                   "outside_set": "CT-KIDNEY Normal-Cyst-Tumor-Stone, stone vs normal, copies removed"})'''

notebook("run_kidney", KID_INTRO, [KID_DATA, KID_RUN],
         {"id": "anshu004/curanex-kidney-stone-swap", "title": "Curanex kidney stone swap", "code_file": "kidney_swap.ipynb",
          "dataset_sources": ["orvile/axial-ct-imaging-dataset-kidney-stone-detection", "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone"]})
