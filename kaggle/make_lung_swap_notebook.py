"""Builds kaggle/lung_ct_swap.ipynb (lung CT, second design). Run from the repo root: python kaggle/make_lung_swap_notebook.py"""

import json
import pathlib

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})


md("""# Curanex: lung CT check, second design (GPU)

The first design (train on IQ-OTH/NCCD, test on the Chest CT-Scan collection) failed: sensitivity 0.284, AUC 0.748.
**Rules fixed before running (2026-11-11):**
- **Train** on the Chest CT-Scan collection `train` folder; best epoch and threshold chosen on its `valid` folder; its `test` folder is reported as an internal check only.
- Binary: cancer (adenocarcinoma, large cell, squamous) against normal. Full fine-tuning of ImageNet EfficientNet-B0, strong augmentation.
- Threshold: highest with validation sensitivity >= 0.85.
- **Outside test, scored once:** IQ-OTH/NCCD (Iraq), never seen by this model. Malignant = positive; benign nodules and normal = negative.
  IQ-OTH near-duplicate slices are counted once, and any slice whose difference hash matches a training image is removed.
- Verified if outside sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off.

**Before running:** Accelerator **GPU T4 x2**, Internet **On**, and add both datasets:
`https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images` and the IQ-OTH/NCCD lung cancer dataset.
Then Run All, and download `lungct_swap.onnx` and `lungct_swap.json` from Output.""")

code("""import glob, hashlib, json, os, random
import numpy as np, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

torch.manual_seed(42); random.seed(42); np.random.seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG = (".jpg", ".jpeg", ".png", ".bmp")
CLASSES = ["Cancer", "Normal"]


def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8)), dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).tobytes()""")

code("""ct = [d for d in glob.glob("/kaggle/input/**/Data", recursive=True) if os.path.isdir(os.path.join(d, "train"))]
assert ct, "Add the Chest CT-Scan images dataset (see the top cell)"
CT = ct[0]
iq = glob.glob("/kaggle/input/**/Malignant cases", recursive=True)
assert iq, "Add the IQ-OTH/NCCD dataset (see the top cell)"
IQ = os.path.dirname(iq[0])


def split(name):
    rows = []
    for f in glob.glob(os.path.join(CT, name, "**", "*"), recursive=True):
        if f.lower().endswith(IMG):
            rows.append((f, 0 if "normal" not in os.path.basename(os.path.dirname(f)).lower() else 1))  # 0 = Cancer, 1 = Normal
    return rows


train, valid, internal = split("train"), split("valid"), split("test")
for n, r in (("train", train), ("valid", valid), ("internal test", internal)):
    print(n, len(r), "cancer:", sum(1 for _, y in r if y == 0))
train_hashes = {dhash(f) for f, _ in train + valid}
outside, seen, dropped = [], set(), 0
for folder in os.listdir(IQ):
    path = os.path.join(IQ, folder)
    if not os.path.isdir(path):
        continue
    label = 1 if folder.lower().startswith("malig") else 0  # 1 = cancer for the outside test
    for f in sorted(glob.glob(os.path.join(path, "*"))):
        if not f.lower().endswith(IMG):
            continue
        h = dhash(f)
        if h in seen:
            continue
        seen.add(h)
        if h in train_hashes:
            dropped += 1
            continue
        outside.append((f, label))
print("outside (IQ-OTH/NCCD):", len(outside), "malignant:", sum(y for _, y in outside), "matches with training removed:", dropped)""")

code("""train_tf = T.Compose([T.Grayscale(3), T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomRotation(15), T.RandomHorizontalFlip(),
                      T.ColorJitter(brightness=0.35, contrast=0.35), T.RandomApply([T.GaussianBlur(5)], p=0.3), T.ToTensor(),
                      T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])  # what the app does


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, y = self.items[i]
        return self.tf(Image.open(f).convert("RGB")), y


loader = lambda rows, tf, shuffle=False: torch.utils.data.DataLoader(DS(rows, tf), batch_size=32, shuffle=shuffle, num_workers=2)
dl_tr, dl_va = loader(train, train_tf, True), loader(valid, eval_tf)
net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
net = net.to(DEV)
yt = np.array([y for _, y in train])
counts = np.bincount(yt, minlength=2)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(yt) / (2 * counts), dtype=torch.float32, device=DEV))
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 20
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)


def predict(dl):
    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.softmax(net(x.to(DEV)), 1).cpu().numpy())
    return np.concatenate(out)


yv = np.array([y for _, y in valid])
best, best_state = -1, None
for ep in range(EPOCHS):
    net.train()
    for x, y in dl_tr:
        opt.zero_grad()
        loss = loss_fn(net(x.to(DEV)), y.to(DEV))
        loss.backward()
        opt.step()
    sched.step()
    auc = roc_auc_score(yv == 0, predict(dl_va)[:, 0])
    print(f"epoch {ep + 1}: validation AUC {auc:.4f}")
    if auc > best:
        best, best_state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)""")

code("""def report(y, s, t):
    f = s >= t
    return {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(f[y == 1].mean()), 3),
            "specificity": round(float((~f[y == 0]).mean()), 3), "auc": round(float(roc_auc_score(y, s)), 3)}


pv = predict(dl_va)[:, 0]
pos = pv[yv == 0]
t_cancer = float(max(v for v in np.unique(pos) if (pos >= v).mean() >= 0.85))
pn = predict(dl_va)[:, 1]
negs = pn[yv == 1]
t_normal = float(max(v for v in np.unique(negs) if (negs >= v).mean() >= 0.85))
yi = np.array([1 - y for _, y in internal])
internal_res = report(yi, predict(loader(internal, eval_tf))[:, 0], t_cancer)
print("internal test (same collection):", internal_res)
yo = np.array([y for _, y in outside])
res = report(yo, predict(loader(outside, eval_tf))[:, 0], t_cancer)
low = min(res["sensitivity"], res["auc"])
res["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
print("OUTSIDE TEST on IQ-OTH/NCCD (once):", res)""")

code("""class Full(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.softmax(self.n(x), 1)


full = Full(net.cpu()).eval()
args = dict(input_names=["image"], output_names=["probs"], opset_version=17, dynamic_axes={"image": {0: "n"}})
try:
    torch.onnx.export(full, torch.randn(1, 3, 224, 224), "/kaggle/working/lungct_swap.onnx", dynamo=False, **args)
except TypeError:
    torch.onnx.export(full, torch.randn(1, 3, 224, 224), "/kaggle/working/lungct_swap.onnx", **args)
meta = {"id": "lungct_swap", "title": "Lung CT (second design)", "labels": CLASSES, "thresholds": [t_cancer, t_normal], "mean": MEAN, "std": STD,
        "validation_auc": round(best, 4), "internal_test": internal_res, "outside_test": res, "images": len(train),
        "onnx_sha256": hashlib.sha256(open("/kaggle/working/lungct_swap.onnx", "rb").read()).hexdigest(),
        "trained_on": "Chest CT-Scan images collection (train folder), full fine-tuning on a Kaggle GPU",
        "outside_set": "IQ-OTH/NCCD (Iraq), malignant vs benign + normal, near-duplicates counted once"}
json.dump(meta, open("/kaggle/working/lungct_swap.json", "w"), indent=1)
print(json.dumps(meta["outside_test"]))""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "lung_ct_swap.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("written", out)
