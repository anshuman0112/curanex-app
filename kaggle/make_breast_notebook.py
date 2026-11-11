"""Builds kaggle/run_breast/breast_us_finetune.ipynb. Run from the repo root: python kaggle/make_breast_notebook.py"""

import json
import pathlib

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})


md("""# Curanex: breast ultrasound, full retraining (GPU)

The live model (frozen features + logistic layer) reached AUC 0.769 on BrEaST (Poland). **Rules fixed before running (2026-11-11),
the same as that attempt except that the whole network is trained:**
- Train on BUS-BRA (Brazil) and BUSI (Egypt) benign and malignant images (masks and normal images left out).
- Validation: 25% of BUS-BRA patients (grouped by case), best epoch by validation AUC; threshold = highest with validation sensitivity >= 0.90.
- **Outside test, scored once:** BrEaST (Poland, TCIA), benign against malignant, official labels.
- Verified if outside sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off.""")

code("""import glob, hashlib, io, json, os, random, zipfile, urllib.request
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

torch.manual_seed(42); random.seed(42); np.random.seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]""")

code("""# BUS-BRA: images + bus_data.csv (ID, Case, Pathology)
csvs = glob.glob("/kaggle/input/**/bus_data.csv", recursive=True)
assert csvs, "BUS-BRA dataset missing"
bra = pd.read_csv(csvs[0])
img_dir = os.path.join(os.path.dirname(csvs[0]), "Images")
rows = []
for _, r in bra.iterrows():
    f = os.path.join(img_dir, f"{r['ID']}.png")
    if os.path.exists(f):
        rows.append((f, int(r["Pathology"] == "malignant"), "bra_" + str(r["Case"]), "bra"))
# BUSI: benign / malignant folders, masks excluded
for cls in ("benign", "malignant"):
    for f in glob.glob(f"/kaggle/input/**/{cls}/*.png", recursive=True):
        if "_mask" not in f:
            rows.append((f, int(cls == "malignant"), "busi_" + os.path.basename(f), "busi"))
print({s: (sum(1 for r in rows if r[3] == s), sum(r[1] for r in rows if r[3] == s)) for s in ("bra", "busi")})
y = np.array([r[1] for r in rows]); groups = np.array([r[2] for r in rows]); src = np.array([r[3] for r in rows])
bra_idx = np.where(src == "bra")[0]
a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(bra_idx, y[bra_idx], groups[bra_idx]))
va = bra_idx[b]
tr = np.setdiff1d(np.arange(len(rows)), va)
print("train", len(tr), "validation", len(va))""")

code("""# BrEaST (Poland) from TCIA, with the official clinical sheet
os.makedirs("/kaggle/working/breast_test", exist_ok=True)
base = "https://www.cancerimagingarchive.net/wp-content/uploads/"
urllib.request.urlretrieve(base + "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip", "/kaggle/working/breast_test/b.zip")
urllib.request.urlretrieve(base + "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx", "/kaggle/working/breast_test/c.xlsx")
zipfile.ZipFile("/kaggle/working/breast_test/b.zip").extractall("/kaggle/working/breast_test")
clin = pd.read_excel("/kaggle/working/breast_test/c.xlsx")
clin = clin[clin["Classification"].isin(["benign", "malignant"])]
outside = [(os.path.join("/kaggle/working/breast_test/BrEaST-Lesions_USG-images_and_masks", f), int(c == "malignant"))
           for f, c in zip(clin["Image_filename"], clin["Classification"])]
print("BrEaST:", len(outside), "malignant:", sum(y for _, y in outside))""")

code("""train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomRotation(15), T.RandomHorizontalFlip(), T.RandomGrayscale(0.5),
                      T.ColorJitter(brightness=0.35, contrast=0.35), T.RandomApply([T.GaussianBlur(5)], p=0.3), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])  # what the app does


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, lab = self.items[i][0], self.items[i][1]
        return self.tf(Image.open(f).convert("RGB")), lab


loader = lambda items, tf, shuffle=False: torch.utils.data.DataLoader(DS(items, tf), batch_size=32, shuffle=shuffle, num_workers=2)
dl_tr, dl_va = loader([rows[i] for i in tr], train_tf, True), loader([rows[i] for i in va], eval_tf)
net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
net = net.to(DEV)
counts = np.bincount(y[tr], minlength=2)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(tr) / (2 * counts), dtype=torch.float32, device=DEV))
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 20
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)


def predict(dl):
    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.softmax(net(x.to(DEV)), 1)[:, 1].cpu().numpy())
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
    auc = roc_auc_score(y[va], predict(dl_va))
    print(f"epoch {ep + 1}: validation AUC {auc:.4f}")
    if auc > best:
        best, best_state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)""")

code("""sv = predict(dl_va)
t = float(max(v for v in np.unique(sv) if (sv[y[va] == 1] >= v).mean() >= 0.90))
yo = np.array([lab for _, lab in outside])
so = predict(loader(outside, eval_tf))
f = so >= t
res = {"n": int(len(yo)), "positives": int(yo.sum()), "sensitivity": round(float(f[yo == 1].mean()), 3),
       "specificity": round(float((~f[yo == 0]).mean()), 3), "auc": round(float(roc_auc_score(yo, so)), 3)}
low = min(res["sensitivity"], res["auc"])
res["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
print("OUTSIDE TEST on BrEaST (once):", res)""")

code("""class Full(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.softmax(self.n(x), 1)


full = Full(net.cpu()).eval()
args = dict(input_names=["image"], output_names=["probs"], opset_version=17, dynamic_axes={"image": {0: "n"}})
try:
    torch.onnx.export(full, torch.randn(1, 3, 224, 224), "/kaggle/working/breast_us_ft.onnx", dynamo=False, **args)
except TypeError:
    torch.onnx.export(full, torch.randn(1, 3, 224, 224), "/kaggle/working/breast_us_ft.onnx", **args)
meta = {"id": "breast_us_ft", "title": "Breast ultrasound (fine-tuned)", "labels": ["benign", "malignant"], "thresholds": [1 - t, t],
        "mean": MEAN, "std": STD, "validation_auc": round(best, 4), "outside_breast_poland": res, "images": int(len(tr)),
        "onnx_sha256": hashlib.sha256(open("/kaggle/working/breast_us_ft.onnx", "rb").read()).hexdigest(),
        "trained_on": f"{len(tr)} ultrasound images from BUS-BRA (Brazil) and BUSI (Egypt), full fine-tuning on a Kaggle GPU"}
json.dump(meta, open("/kaggle/working/breast_us_ft.json", "w"), indent=1)
import shutil; shutil.rmtree("/kaggle/working/breast_test", ignore_errors=True)
print(json.dumps(res))""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "run_breast" / "breast_us_finetune.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
(out.parent / "kernel-metadata.json").write_text(json.dumps({
    "id": "anshu004/curanex-breast-ultrasound-finetune", "title": "Curanex breast ultrasound finetune", "code_file": "breast_us_finetune.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["orvile/bus-bra-a-breast-ultrasound-dataset", "aryashah2k/breast-ultrasound-images-dataset"],
    "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", out)
