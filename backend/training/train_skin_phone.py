"""Skin check retrained for phone photos. From backend/: python training/train_skin_phone.py

The dermatoscope-only model found 44% of melanomas on phone photos (PAD-UFES-20), so phone images now join training.
Rules fixed first: PAD-UFES-20 is split by patient 60/20/20 and no patient crosses a split; HAM10000 joins training only;
C and threshold come from validation phone photos, threshold the highest with sensitivity >= 0.85; phone test patients are
scored once. model.onnx is replaced only if test sensitivity and AUC reach 0.80. Otherwise the old model and Unverified stay.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from tools.evaluate_outside import report  # noqa: E402
from training.train_skin import CACHE, backbone, prep  # noqa: E402

ROOT = BACKEND / "data" / "eval" / "pad"
FEATS = BACKEND / "data" / "eval" / "features_pad.npz"
OUT = BACKEND / "workers" / "torch" / "models" / "skin_lesion_check"
TARGET = 0.85


def main() -> None:
    torch.set_num_threads(2)
    table = next(ROOT.rglob("metadata.csv"))
    files = {f.name: f for f in ROOT.rglob("*.png")}
    with table.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["img_id"] in files]
    y = np.array([int(r["diagnostic"] in ("MEL", "BCC", "SCC", "ACK")) for r in rows])
    kind = np.array([r["diagnostic"] for r in rows])
    group = np.array([r["patient_id"] for r in rows])
    net = backbone()
    if FEATS.exists():
        X = np.load(FEATS)["X"]
    else:
        part = FEATS.with_suffix(".part.npz")
        out, start = [], 0
        if part.exists():
            d = np.load(part)
            out, start = [d["X"]], int(d["next"])
        with torch.no_grad():
            for i in range(start, len(rows), 24):
                batch = torch.from_numpy(np.stack([prep(files[r["img_id"]].read_bytes()) for r in rows[i:i + 24]]))
                out.append(net(batch).numpy())
                if i % 480 == 0:
                    print("features", i, flush=True)
                    np.savez(part, X=np.concatenate(out), next=i + 24)
        X = np.concatenate(out)
        np.savez(FEATS, X=X)
        part.unlink(missing_ok=True)
    tr, rest = next(GroupShuffleSplit(1, test_size=0.40, random_state=42).split(X, y, group))
    a, b = next(GroupShuffleSplit(1, test_size=0.50, random_state=42).split(rest, y[rest], group[rest]))
    va, te = rest[a], rest[b]
    assert not set(group[tr]) & set(group[te])
    import zipfile
    z = zipfile.ZipFile(BACKEND / "data" / "eval" / "ham" / "data.zip")
    names = {Path(n).stem for n in z.namelist() if n.lower().endswith(".jpg") and n.startswith("HAM10000_images")}
    ham = [ln.split(",") for ln in z.read("HAM10000_metadata.csv").decode().splitlines()[1:] if ln]
    ham = [r for r in ham if r[1] in names]
    Xh = np.load(CACHE)["X"]
    yh = np.array([int(r[2] in {"mel", "bcc", "akiec"}) for r in ham])
    Xt, yt = np.concatenate([X[tr], Xh]), np.concatenate([y[tr], yh])
    w = np.concatenate([np.full(len(tr), 3.0), np.ones(len(yh))])  # phone photos are the target, so they count for more
    scaler = StandardScaler().fit(Xt)
    best = None
    for c in (0.001, 0.01, 0.1):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=4000).fit(scaler.transform(Xt), yt, sample_weight=w)
        s = roc_auc_score(y[va], m.predict_proba(scaler.transform(X[va]))[:, 1])
        print(f"C={c} phone validation AUC {s:.4f}", flush=True)
        if best is None or s > best[0]:
            best = (s, c, m)
    _, c, model = best
    score = lambda Z: model.predict_proba(scaler.transform(Z))[:, 1]  # noqa: E731
    sv = score(X[va])
    t = float(max(v for v in np.unique(sv) if (sv[y[va] == 1] >= v).mean() >= TARGET))
    st = score(X[te])
    result = {"C": c, "threshold": t, "phone_test": report(y[te], st, t), "found_by_type": {d: round(float((st[kind[te] == d] >= t).mean()), 3) for d in sorted(set(kind[te]))}}
    print(json.dumps(result), flush=True)
    r = result["phone_test"]
    if r["sensitivity"] < 0.80 or r["auc"] < 0.80:
        print("FAILED: model left unchanged", flush=True)
        (BACKEND / "data" / "eval" / "skin_phone_result.json").write_text(json.dumps(result, indent=1))
        return

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

    sample = torch.from_numpy(prep(files[rows[0]["img_id"]].read_bytes())[None])
    path = OUT / "model.onnx"
    torch.onnx.export(Full().eval(), (sample,), str(path), input_names=["image"], output_names=["score"], opset_version=17, dynamo=False, dynamic_axes={"image": {0: "n"}})
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = json.loads((OUT / "model_meta.json").read_text())
    meta.update({"threshold": t, "phone": result, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=1))
    print("PASSED: model replaced", flush=True)


if __name__ == "__main__":
    main()
