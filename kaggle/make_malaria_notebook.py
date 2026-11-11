"""Builds kaggle/run_malaria/malaria_finetune.ipynb. Run from the repo root: python kaggle/make_malaria_notebook.py"""

import json
import pathlib

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})


md("""# Curanex: malaria cell check, full retraining (GPU)

The live model (frozen features) failed the BBBC041 outside test (AUC 0.661). **Rules fixed before running (2026-11-11):**
- Train on the NIH malaria cell images (Parasitized vs Uninfected); validation = 15% of patients (patient code from the file name).
- Full fine-tuning of ImageNet EfficientNet-B0 with strong colour augmentation (hue, saturation, brightness, contrast), because stain colour is what differs between labs.
- Best epoch by validation AUC; threshold = highest with validation sensitivity >= 0.85.
- **Outside test, scored once:** BBBC041 (P. vivax, different lab, stain and cameras). Cells cropped from the published boxes; infected = ring,
  trophozoite, schizont or gametocyte; uninfected = red blood cell; 800 of each sampled with seed 42 (same as the earlier test).
- Verified if outside sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off.""")

code("""import glob, hashlib, json, os, random
import numpy as np, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

torch.manual_seed(42); random.seed(42); np.random.seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["Parasitized", "Uninfected"]  # same labels as the live model""")

code("""rows = []
for i, c in enumerate(CLASSES):
    for f in glob.glob(f"/kaggle/input/**/cell_images/{c}/*.png", recursive=True):
        if "/cell_images/cell_images/" in f:
            continue  # the dataset ships a duplicate copy of every image
        rows.append((f, i, os.path.basename(f).split("_")[0]))
assert rows, "NIH malaria cell images missing"
y = np.array([r[1] for r in rows]); groups = np.array([r[2] for r in rows])
tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=42).split(rows, y, groups))
print("cells", len(rows), "patients", len(set(groups)), "train", len(tr), "validation", len(va))

# BBBC041 outside cells
tables = glob.glob("/kaggle/input/**/malaria/training.json", recursive=True) + glob.glob("/kaggle/input/**/malaria/test.json", recursive=True)
assert tables, "BBBC041 (malaria bounding boxes) missing"
root = os.path.dirname(tables[0])
INFECTED = {"ring", "trophozoite", "schizont", "gametocyte"}
boxes = {0: [], 1: []}
for t in tables:
    for e in json.load(open(t)):
        name = os.path.join(root, "images", os.path.basename(e["image"]["pathname"]))
        for b in e["objects"]:
            if b["category"] == "red blood cell":
                boxes[0].append((name, b["bounding_box"]))
            elif b["category"] in INFECTED:
                boxes[1].append((name, b["bounding_box"]))
rng = random.Random(42)
pick = [(n, b, lab) for lab in (0, 1) for n, b in rng.sample(boxes[lab], min(800, len(boxes[lab])))]
pick.sort(key=lambda r: r[0])
print("BBBC041 picked", len(pick), {k: len(v) for k, v in boxes.items()})""")

code("""train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomRotation(180), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.15), T.RandomApply([T.GaussianBlur(5)], p=0.3),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])  # what the app does


class Cells(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(Image.open(self.items[i][0]).convert("RGB")), self.items[i][1]


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, b, lab = self.items[i]
        lo, hi = b["minimum"], b["maximum"]
        img = Image.open(name).convert("RGB").crop((lo["c"], lo["r"], hi["c"], hi["r"]))
        return self.tf(img), lab


dl_tr = torch.utils.data.DataLoader(Cells([rows[i] for i in tr], train_tf), batch_size=64, shuffle=True, num_workers=2)
dl_va = torch.utils.data.DataLoader(Cells([rows[i] for i in va], eval_tf), batch_size=128, num_workers=2)
net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
net = net.to(DEV)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 6
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)


def predict(dl):
    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.softmax(net(x.to(DEV)), 1).cpu().numpy())
    return np.concatenate(out)


best, best_state = -1, None
for ep in range(EPOCHS):
    net.train()
    for x, lab in dl_tr:
        opt.zero_grad()
        loss = loss_fn(net(x.to(DEV)), lab.to(DEV))
        loss.backward()
        opt.step()
    sched.step()
    auc = roc_auc_score(y[va] == 0, predict(dl_va)[:, 0])
    print(f"epoch {ep + 1}: validation AUC {auc:.4f}")
    if auc > best:
        best, best_state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)""")

code("""pv = predict(dl_va)
pos = pv[y[va] == 0, 0]
t_par = float(max(v for v in np.unique(pos) if (pos >= v).mean() >= 0.85))
neg = pv[y[va] == 1, 1]
t_un = float(max(v for v in np.unique(neg) if (neg >= v).mean() >= 0.85))
po = predict(torch.utils.data.DataLoader(Crops(pick, eval_tf), batch_size=128, num_workers=2))[:, 0]
yo = np.array([lab for _, _, lab in pick])
f = po >= t_par
res = {"n": int(len(yo)), "positives": int(yo.sum()), "sensitivity": round(float(f[yo == 1].mean()), 3),
       "specificity": round(float((~f[yo == 0]).mean()), 3), "auc": round(float(roc_auc_score(yo, po)), 3)}
low = min(res["sensitivity"], res["auc"])
res["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
print("OUTSIDE TEST on BBBC041 (once):", res)""")

code("""class Full(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.softmax(self.n(x), 1)


full = Full(net.cpu()).eval()
args = dict(input_names=["image"], output_names=["probs"], opset_version=17, dynamic_axes={"image": {0: "n"}})
try:
    torch.onnx.export(full, torch.randn(1, 3, 224, 224), "/kaggle/working/malaria_ft.onnx", dynamo=False, **args)
except TypeError:
    torch.onnx.export(full, torch.randn(1, 3, 224, 224), "/kaggle/working/malaria_ft.onnx", **args)
meta = {"id": "malaria_ft", "title": "Malaria cell (fine-tuned)", "labels": CLASSES, "thresholds": [t_par, t_un], "mean": MEAN, "std": STD,
        "validation_auc": round(best, 4), "outside_test": res, "images": int(len(tr)),
        "onnx_sha256": hashlib.sha256(open("/kaggle/working/malaria_ft.onnx", "rb").read()).hexdigest(),
        "trained_on": "NIH malaria cell images (Bangladesh), split by patient, full fine-tuning on a Kaggle GPU"}
json.dump(meta, open("/kaggle/working/malaria_ft.json", "w"), indent=1)
print(json.dumps(res))""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "run_malaria" / "malaria_finetune.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
(out.parent / "kernel-metadata.json").write_text(json.dumps({
    "id": "anshu004/curanex-malaria-finetune", "title": "Curanex malaria finetune", "code_file": "malaria_finetune.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["iarunava/cell-images-for-detecting-malaria", "kmader/malaria-bounding-boxes"],
    "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", out)
