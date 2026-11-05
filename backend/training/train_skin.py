"""Skin lesion check on HAM10000, trained by the Curanex team. Run from backend/: python training/train_skin.py

Pretrained EfficientNet-B0 features plus a logistic layer, which is fast with no GPU.
Rules fixed before scoring: suspicious means melanoma, basal cell carcinoma or actinic keratosis; split 70/15/15 by lesion so
one lesion never sits on both sides; threshold is the highest with validation sensitivity >= 0.85; test scored once.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND / "data" / "eval" / "ham"
CACHE = ROOT / "features.npz"
OUT = BACKEND / "workers" / "torch" / "models" / "skin_lesion_check"
SUSPICIOUS = {"mel", "bcc", "akiec"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def prep(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    s = 256 / min(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    a = np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255
    return ((a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)).transpose(2, 0, 1)


def backbone():
    return nn.Sequential(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features, nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()


def report(y, s, t):
    tn, fp, fn, tp = confusion_matrix(y, s >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, s)), 3)}


def main() -> None:
    torch.set_num_threads(3)
    z = zipfile.ZipFile(ROOT / "data.zip")
    names = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg") and n.startswith("HAM10000_images")}
    rows = [ln.split(",") for ln in z.read("HAM10000_metadata.csv").decode().splitlines()[1:] if ln]
    rows = [r for r in rows if r[1] in names]
    y = np.array([int(r[2] in SUSPICIOUS) for r in rows])
    groups = np.array([r[0] for r in rows])
    kinds = np.array([r[2] for r in rows])
    print(len(rows), int(y.sum()), flush=True)
    net = backbone()
    if CACHE.exists():
        X = np.load(CACHE)["X"]
    else:
        X = []
        with torch.no_grad():
            for i in range(0, len(rows), 32):
                batch = torch.from_numpy(np.stack([prep(z.read(names[r[1]])) for r in rows[i:i + 32]]))
                X.append(net(batch).numpy())
                if i % 960 == 0:
                    print("features", i, flush=True)
        X = np.concatenate(X)
        np.savez(CACHE, X=X)
    tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=42).split(X, y, groups))
    va_i, te_i = next(GroupShuffleSplit(1, test_size=0.50, random_state=42).split(rest, y[rest], groups[rest]))
    va, te = rest[va_i], rest[te_i]
    assert not set(groups[tr]) & set(groups[te])
    scaler = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=4000).fit(scaler.transform(X[tr]), y[tr])
        a = roc_auc_score(y[va], m.predict_proba(scaler.transform(X[va]))[:, 1])
        print(f"C={c} val AUC {a:.4f}", flush=True)
        if best is None or a > best[0]:
            best = (a, c, m)
    _, c, model = best
    score = lambda Z: model.predict_proba(scaler.transform(Z))[:, 1]  # noqa: E731
    t = float(max(v for v in np.unique(score(X[va])) if (score(X[va])[y[va] == 1] >= v).mean() >= 0.85))
    s_te = score(X[te])
    result = {"C": c, "threshold": t, "test": report(y[te], s_te, t), "melanoma_found": round(float((s_te[kinds[te] == "mel"] >= t).mean()), 3), "training_images": len(tr),
              "suspicious": sorted(SUSPICIOUS), "mean": MEAN, "std": STD}
    print(json.dumps({k: v for k, v in result.items() if k not in ("mean", "std")}), flush=True)

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))
            self.register_buffer("coef", torch.tensor(model.coef_[0], dtype=torch.float32))
            self.intercept = float(model.intercept_[0])

        def forward(self, x):
            return torch.sigmoid((((self.net(x) - self.mean) / self.scale) * self.coef).sum(1) + self.intercept)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "model.onnx"
    sample = torch.from_numpy(np.stack([prep(z.read(names[rows[0][1]]))]))
    full = Full().eval()
    torch.onnx.export(full, (sample,), str(path), input_names=["image"], output_names=["score"], opset_version=17, dynamo=False, dynamic_axes={"image": {0: "n"}})
    import hashlib

    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        result["onnx_diff"] = float(np.abs(s.run(None, {"image": sample.numpy()})[0] - full(sample).numpy()).max())
    assert result["onnx_diff"] < 1e-3
    result["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "model_meta.json").write_text(json.dumps(result, indent=1))
    print("wrote", path.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
