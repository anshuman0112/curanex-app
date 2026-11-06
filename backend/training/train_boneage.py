"""Bone age from a child's hand X-ray, trained by the Curanex team on RSNA Bone Age. From backend/: python training/train_boneage.py

Rules fixed first (owner decision: pass is a mean error of 12 months or less, worded "estimated bone age"): 70/15/15 split;
ridge strength chosen on validation; test scored once. EfficientNet-B0 features plus ridge regression, so no GPU is needed.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from training.train_skin import MEAN, STD, backbone  # noqa: E402

ZIP = BACKEND / "data" / "eval" / "boneage" / "data.zip"
FEATS = BACKEND / "data" / "eval" / "features_boneage.npz"
OUT = BACKEND / "workers" / "torch" / "models" / "multi"
LIMIT = 6000


def prep(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255
    return ((a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)).transpose(2, 0, 1)


def main() -> None:
    torch.set_num_threads(3)
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    table = next(n for n in names if n.endswith("boneage-training-dataset.csv"))
    files = {Path(n).stem: n for n in names if n.endswith(".png") and "training" in n}
    rows = [ln.split(",") for ln in z.read(table).decode().splitlines()[1:] if ln]
    rows = [r for r in rows if r[0] in files]
    np.random.default_rng(42).shuffle(rows)
    rows = rows[:LIMIT]
    y = np.array([float(r[1]) for r in rows])
    male = np.array([float(r[2].strip().lower() == "true") for r in rows])
    print(len(rows), "images, mean age", round(float(y.mean()), 1), flush=True)
    net = backbone()
    part = FEATS.with_suffix(".part.npz")
    if FEATS.exists():
        X = np.load(FEATS)["X"]
    else:
        out, start = [], 0
        if part.exists():
            d = np.load(part)
            out, start = [d["X"]], int(d["next"])
        with torch.no_grad():
            for i in range(start, len(rows), 24):
                batch = torch.from_numpy(np.stack([prep(z.read(files[r[0]])) for r in rows[i:i + 24]]))
                out.append(net(batch).numpy())
                if i % 480 == 0:
                    print("features", i, flush=True)
                    np.savez(part, X=np.concatenate(out), next=i + 24)
        X = np.concatenate(out)
        np.savez(FEATS, X=X)
        part.unlink(missing_ok=True)
    X2 = np.c_[X, male]
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, random_state=42)
    va, te = train_test_split(rest, test_size=0.50, random_state=42)
    sc = StandardScaler().fit(X2[tr])
    best = None
    for a in (10, 100, 1000, 10000):
        m = Ridge(alpha=a).fit(sc.transform(X2[tr]), y[tr])
        e = float(np.abs(m.predict(sc.transform(X2[va])) - y[va]).mean())
        print(f"alpha={a} val MAE {e:.2f} months", flush=True)
        if best is None or e < best[0]:
            best = (e, a, m)
    _, a, model = best
    mae = float(np.abs(model.predict(sc.transform(X2[te])) - y[te]).mean())
    result = {"id": "boneage", "alpha": a, "test_mae_months": round(mae, 2), "n": len(te), "passed": mae <= 12, "mean": MEAN, "std": STD, "age_range": [float(y.min()), float(y.max())]}
    print(json.dumps(result), flush=True)
    if mae > 12:
        print("FAILED: not exported", flush=True)
        return

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(sc.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(sc.scale_, dtype=torch.float32))
            self.register_buffer("coef", torch.tensor(model.coef_, dtype=torch.float32))
            self.intercept = float(model.intercept_)

        def forward(self, x, male):
            z_ = (torch.cat([self.net(x), male], 1) - self.mean) / self.scale
            return (z_ * self.coef).sum(1) + self.intercept

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "boneage.onnx"
    sample = torch.from_numpy(prep(z.read(files[rows[0][0]]))[None])
    torch.onnx.export(Full().eval(), (sample, torch.ones(1, 1)), str(path), input_names=["image", "male"], output_names=["months"], opset_version=17, dynamo=False)
    result["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "boneage.json").write_text(json.dumps(result, indent=1))
    print("wrote boneage", flush=True)


if __name__ == "__main__":
    main()
