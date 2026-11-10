"""Builds kaggle/lung_ct_finetune.ipynb. Run from the repo root: python kaggle/make_lung_notebook.py"""

import json
import pathlib

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})


md("""# Curanex: lung CT check, full retraining (GPU)

Trained by the Curanex team. **Rules fixed before running (2026-11-10):**
- Train on IQ-OTH/NCCD (Iraq), near-duplicate slices removed (difference hash), 85/15 stratified split for validation.
- Full fine-tuning of ImageNet EfficientNet-B0 with strong augmentation (crops, rotation, brightness/contrast, blur) so it depends less on one scanner.
- Best epoch by validation macro AUC; per-class threshold = highest with validation sensitivity >= 0.85 (same rule as before).
- **Outside test, scored once:** the separate Chest CT-Scan collection (all 1,000 images), cancer (adenocarcinoma, large cell, squamous) against normal.
- Verified only if outside sensitivity and AUC are both >= 0.80; live at >= 0.75.

**Before running:** Settings -> Accelerator -> **GPU T4 x2**; Internet **On**. Add data (right panel -> Add Input) and search for:
1. `the iqothnccd lung cancer dataset`
2. `chest ctscan images` (by mohamedhanyyy)

Then Run All. When it finishes, download `lungct_ft.onnx` and `lungct_ft.json` from the Output panel.""")

code("""import glob, hashlib, json, os, random
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
IMG = (".jpg", ".jpeg", ".png", ".bmp")""")

code("""# Find the two datasets wherever Kaggle mounted them
iq = glob.glob("/kaggle/input/**/Malignant cases", recursive=True)
assert iq, "Add the IQ-OTH/NCCD dataset (see the top cell)"
IQ = os.path.dirname(iq[0])
ct = [d for d in glob.glob("/kaggle/input/**/Data", recursive=True) if os.path.isdir(os.path.join(d, "test"))]
assert ct, "Add the Chest CT-Scan images dataset (see the top cell)"
CT = ct[0]
CLASSES = ["Bengin", "Malignant", "Normal"]  # same labels as the live model


def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8)), dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).tobytes()


rows, seen = [], set()
for i, c in enumerate(CLASSES):
    folder = [d for d in os.listdir(IQ) if d.lower().startswith(c.lower()[:5])][0]
    for f in sorted(glob.glob(os.path.join(IQ, folder, "*"))):
        if not f.lower().endswith(IMG):
            continue
        h = dhash(f)
        if h in seen:
            continue  # near-duplicate slice
        seen.add(h)
        rows.append((f, i))
print("IQ-OTH/NCCD after removing near-duplicates:", {c: sum(1 for _, y in rows if y == i) for i, c in enumerate(CLASSES)})
outside = [(f, int("normal" not in os.path.basename(os.path.dirname(f)).lower()))
           for f in glob.glob(os.path.join(CT, "**", "*"), recursive=True) if f.lower().endswith(IMG)]
print("outside images:", len(outside), "cancer:", sum(y for _, y in outside))
y_all = np.array([y for _, y in rows])
tr, va = train_test_split(np.arange(len(rows)), test_size=0.15, random_state=42, stratify=y_all)""")

code("""train_tf = T.Compose([T.Grayscale(3), T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomRotation(15), T.RandomHorizontalFlip(),
                      T.ColorJitter(brightness=0.35, contrast=0.35), T.RandomApply([T.GaussianBlur(5)], p=0.3), T.ToTensor(),
                      T.Normalize(MEAN, STD)])
# evaluation = exactly what the app does: shortest side 256, centre crop 224, ImageNet normalisation
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, y = self.items[i]
        return self.tf(Image.open(f).convert("RGB")), y


dl_tr = torch.utils.data.DataLoader(DS([rows[i] for i in tr], train_tf), batch_size=32, shuffle=True, num_workers=2)
dl_va = torch.utils.data.DataLoader(DS([rows[i] for i in va], eval_tf), batch_size=64, num_workers=2)
net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(CLASSES))
net = net.to(DEV)
counts = np.bincount(y_all[tr], minlength=3)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(tr) / (3 * counts), dtype=torch.float32, device=DEV))
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 15
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)


def predict(loader):
    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.softmax(net(x.to(DEV)), 1).cpu().numpy())
    return np.concatenate(out)


best, best_state = -1, None
for ep in range(EPOCHS):
    net.train()
    for x, y in dl_tr:
        opt.zero_grad()
        loss = loss_fn(net(x.to(DEV)), y.to(DEV))
        loss.backward()
        opt.step()
    sched.step()
    pv = predict(dl_va)
    auc = roc_auc_score(y_all[va], pv, multi_class="ovr", average="macro")
    print(f"epoch {ep + 1}: validation macro AUC {auc:.4f}")
    if auc > best:
        best, best_state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)""")

code("""pv = predict(dl_va)
thresholds = []
for i in range(3):
    pos = pv[y_all[va] == i, i]
    thresholds.append(float(max(v for v in np.unique(pos) if (pos >= v).mean() >= 0.85)))
# Outside test, scored once: cancer = the Malignant class reaching its threshold
dl_out = torch.utils.data.DataLoader(DS(outside, eval_tf), batch_size=64, num_workers=2)
po = predict(dl_out)
yo = np.array([y for _, y in outside])
m = CLASSES.index("Malignant")
flag = po[:, m] >= thresholds[m]
res = {"n": int(len(yo)), "positives": int(yo.sum()), "sensitivity": round(float(flag[yo == 1].mean()), 3),
       "specificity": round(float((~flag[yo == 0]).mean()), 3), "auc": round(float(roc_auc_score(yo, po[:, m])), 3)}
low = min(res["sensitivity"], res["auc"])
res["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
print("OUTSIDE TEST (once):", res)""")

code("""# Export in the format the Curanex image worker reads (softmax over the three classes)
class Full(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.softmax(self.n(x), 1)


full = Full(net.cpu()).eval()
args = dict(input_names=["image"], output_names=["probs"], opset_version=17, dynamic_axes={"image": {0: "n"}})
try:
    torch.onnx.export(full, torch.randn(1, 3, 224, 224), "/kaggle/working/lungct_ft.onnx", dynamo=False, **args)
except TypeError:  # older PyTorch has no dynamo switch
    torch.onnx.export(full, torch.randn(1, 3, 224, 224), "/kaggle/working/lungct_ft.onnx", **args)
meta = {"id": "lungct_ft", "title": "Lung CT (fine-tuned)", "labels": CLASSES, "thresholds": thresholds, "mean": MEAN, "std": STD,
        "validation_macro_auc": round(best, 4), "outside_test": res, "images": len(rows),
        "onnx_sha256": hashlib.sha256(open("/kaggle/working/lungct_ft.onnx", "rb").read()).hexdigest(),
        "trained_on": "IQ-OTH/NCCD (Iraq), near-duplicates removed, full fine-tuning on a Kaggle GPU",
        "outside_set": "Chest CT-Scan images collection, all 1,000 images, cancer vs normal"}
json.dump(meta, open("/kaggle/working/lungct_ft.json", "w"), indent=1)
print(json.dumps(meta["outside_test"]))""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "lung_ct_finetune.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("written", out)
