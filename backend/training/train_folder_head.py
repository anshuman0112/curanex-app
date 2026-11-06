"""One image model covering many diseases, trained by the Curanex team. From backend/: python training/train_folder_head.py <name> <folder> "<title>"

<folder> holds one sub-folder per class, or train/ and test/ folders that each do. Pretrained EfficientNet-B0 features plus a
logistic layer, so it runs on a laptop with no GPU.
Rules fixed before scoring: exact duplicate files are removed first, so a copy can't sit on both sides; a dataset's own test
split is kept if it has one, otherwise 70/15/15 stratified; each class gets a threshold from validation, the highest with
sensitivity >= 0.85; the test set is scored once. A class counts as passed only with sensitivity >= 0.80 and AUC >= 0.80.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

BACKEND = Path(__file__).resolve().parents[1]
OUT = BACKEND / "workers" / "torch" / "models" / "multi"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMAGE = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MAX_PER_CLASS = 1500  # keeps very large classes from dominating CPU time
NEAR_DUPLICATES = os.getenv("NEAR_DUPLICATES") == "1"  # for sets known to contain augmented copies
EXCLUDE = os.getenv("EXCLUDE_NAME", "")  # for example segmentation masks stored beside the images


def prep(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    s = 256 / min(img.size)
    img = img.resize((max(224, round(img.width * s)), max(224, round(img.height * s))), Image.BILINEAR)
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    a = np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255
    return ((a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)).transpose(2, 0, 1)


def gather(root: Path):
    splits = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    train_dir = next((splits[k] for k in ("train", "training") if k in splits), None)
    test_dir = next((splits[k] for k in ("test", "testing") if k in splits), None)
    sets = {"train": train_dir, "test": test_dir} if train_dir and test_dir else {"all": root}
    seen, rows = set(), []
    for split, base in sets.items():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            files = sorted(f for f in d.rglob("*") if f.suffix.lower() in IMAGE and not (EXCLUDE and EXCLUDE in f.name))
            kept = 0
            for f in files:
                h = hashlib.md5(f.read_bytes()).hexdigest()  # noqa: S324 finds duplicates, not security
                if h in seen:
                    continue
                seen.add(h)
                if NEAR_DUPLICATES:  # small differences hash alike, which catches crops and brightness copies
                    try:
                        a = np.asarray(Image.open(f).convert("L").resize((9, 8)), dtype=np.float32)
                    except OSError:
                        continue
                    near = (a[:, 1:] > a[:, :-1]).tobytes()
                    if near in seen:
                        continue
                    seen.add(near)
                rows.append((f, d.name, split))
                kept += 1
                if kept >= MAX_PER_CLASS:
                    break
    return rows


def main(name: str, folder: str, title: str) -> None:
    torch.set_num_threads(3)
    rows = gather(Path(folder))
    classes = sorted({r[1] for r in rows})
    y = np.array([classes.index(r[1]) for r in rows])
    split = np.array([r[2] for r in rows])
    print(name, len(rows), "images", len(classes), "classes", flush=True)
    net = nn.Sequential(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features, nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    cache = BACKEND / "data" / "eval" / f"features_{name}.npz"
    if cache.exists():
        X, keep = np.load(cache)["X"], np.load(cache)["keep"]
    else:
        part = cache.with_suffix(".part.npz")
        X, keep, start = [], [], 0
        if part.exists():  # a restart loses at most the last few hundred images
            d = np.load(part)
            X, keep, start = [d["X"]], d["keep"].tolist(), int(d["next"])
            print("resuming at", start, flush=True)
        with torch.no_grad():
            for i in range(start, len(rows), 24):
                batch, ok = [], []
                for j in range(i, min(i + 24, len(rows))):
                    try:
                        batch.append(prep(rows[j][0]))
                        ok.append(j)
                    except OSError:
                        continue
                if batch:
                    X.append(net(torch.from_numpy(np.stack(batch))).numpy())
                    keep += ok
                if i % 480 == 0:
                    print("features", i, flush=True)
                    np.savez(part, X=np.concatenate(X), keep=np.array(keep), next=i + 24)
        X, keep = np.concatenate(X), np.array(keep)
        np.savez(cache, X=X, keep=keep)
        part.unlink(missing_ok=True)
    y, split = y[keep], split[keep]
    idx = np.arange(len(y))
    if (split == "test").any():
        te = idx[split == "test"]
        tr, va = train_test_split(idx[split != "test"], test_size=0.15, random_state=42, stratify=y[split != "test"])
    else:
        tr, rest = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
        va, te = train_test_split(rest, test_size=0.50, random_state=42, stratify=y[rest])
    scaler = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(scaler.transform(X[tr]), y[tr])
        p = m.predict_proba(scaler.transform(X[va]))
        a = roc_auc_score(y[va], p[:, 1]) if len(classes) == 2 else roc_auc_score(y[va], p, multi_class="ovr", average="macro")
        print(f"C={c} val AUC {a:.4f}", flush=True)
        if best is None or a > best[0]:
            best = (a, c, m)
    _, c, model = best
    pv, pt = model.predict_proba(scaler.transform(X[va])), model.predict_proba(scaler.transform(X[te]))
    per, passed, thresholds = {}, [], []
    for i, label in enumerate(classes):
        pos = pv[y[va] == i, i]
        t = float(max(v for v in np.unique(pos) if (pos >= v).mean() >= 0.85))
        thresholds.append(t)
        truth, s = y[te] == i, pt[:, i]
        sens, spec = float((s[truth] >= t).mean()), float((s[~truth] < t).mean())
        auc = float(roc_auc_score(truth, s))
        per[label] = {"n": int(truth.sum()), "sensitivity": round(sens, 3), "specificity": round(spec, 3), "auc": round(auc, 3)}
        if sens >= 0.80 and auc >= 0.80:
            passed.append(label)
    result = {"id": name, "title": title, "labels": classes, "images": len(y), "c": c, "accuracy": round(float(accuracy_score(y[te], pt.argmax(1))), 3), "per_class": per, "passed": passed,
              "thresholds": thresholds, "mean": MEAN, "std": STD}
    print(json.dumps({k: result[k] for k in ("id", "images", "accuracy", "passed")}), flush=True)
    for label, v in per.items():
        print(f"  {label}: {v}", flush=True)

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))
            coef = model.coef_ if len(classes) > 2 else np.vstack([-model.coef_, model.coef_])
            icpt = model.intercept_ if len(classes) > 2 else np.array([-model.intercept_[0], model.intercept_[0]])
            self.register_buffer("coef", torch.tensor(coef, dtype=torch.float32))
            self.register_buffer("icpt", torch.tensor(icpt, dtype=torch.float32))

        def forward(self, x):
            return (((self.net(x) - self.mean) / self.scale) @ self.coef.T + self.icpt).softmax(1)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    sample = torch.from_numpy(prep(rows[keep[0]][0])[None])
    torch.onnx.export(Full().eval(), (sample,), str(path), input_names=["image"], output_names=["probs"], opset_version=17, dynamo=False, dynamic_axes={"image": {0: "n"}})
    result["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    print("wrote", name, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
