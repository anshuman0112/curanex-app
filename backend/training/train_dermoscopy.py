"""Dermoscopy skin lesion check (clinic-grade), trained by the Curanex team. From backend/: python training/train_dermoscopy.py

Exactly the recipe and split of training/train_skin.py, whose HAM10000 test was scored on 2026-11-06 (sensitivity 0.855,
specificity 0.741, AUC 0.882). That model was later replaced in the app by a phone-photo version; this script regenerates
the dermoscopy model as its own clinic model and checks that the same test numbers come back, so the test set is not reused
to make any new choice. Exported in the MultiHeadAdapter format.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from training.train_skin import CACHE, MEAN, ROOT, STD, SUSPICIOUS, backbone, prep, report  # noqa: E402

OUT = BACKEND / "workers" / "torch" / "models" / "multi"
RECORDED = {"sensitivity": 0.855, "specificity": 0.741, "auc": 0.882}


def main() -> None:
    z = zipfile.ZipFile(ROOT / "data.zip")
    names = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg") and n.startswith("HAM10000_images")}
    rows = [ln.split(",") for ln in z.read("HAM10000_metadata.csv").decode().splitlines()[1:] if ln]
    rows = [r for r in rows if r[1] in names]
    y = np.array([int(r[2] in SUSPICIOUS) for r in rows])
    groups = np.array([r[0] for r in rows])
    X = np.load(CACHE)["X"]
    tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=42).split(X, y, groups))
    va_i, te_i = next(GroupShuffleSplit(1, test_size=0.50, random_state=42).split(rest, y[rest], groups[rest]))
    va, te = rest[va_i], rest[te_i]
    scaler = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=4000).fit(scaler.transform(X[tr]), y[tr])
        a = roc_auc_score(y[va], m.predict_proba(scaler.transform(X[va]))[:, 1])
        if best is None or a > best[0]:
            best = (a, c, m)
    _, c, model = best
    score = lambda Z: model.predict_proba(scaler.transform(Z))[:, 1]  # noqa: E731
    sv = score(X[va])
    t = float(max(v for v in np.unique(sv) if (sv[y[va] == 1] >= v).mean() >= 0.85))
    test = report(y[te], score(X[te]), t)
    print("reproduced:", test, flush=True)
    assert all(abs(test[k] - v) < 0.002 for k, v in RECORDED.items()), "does not match the recorded 2026-11-06 test"
    net = backbone()

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))
            self.register_buffer("coef", torch.tensor(model.coef_[0], dtype=torch.float32))
            self.intercept = float(model.intercept_[0])

        def forward(self, x):
            zz = (((self.net(x) - self.mean) / self.scale) * self.coef).sum(1) + self.intercept
            return torch.stack([-zz, zz], 1).mul(0.5).softmax(1)  # [not suspicious, suspicious]

    path = OUT / "dermoscopy.onnx"
    sample = torch.from_numpy(prep(z.read(names[rows[0][1]]))[None])
    torch.onnx.export(Full().eval(), (sample,), str(path), input_names=["image"], output_names=["probs"], opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "n"}})
    result = {"id": "dermoscopy", "title": "Dermoscopy skin lesion", "labels": ["not_suspicious", "suspicious"], "thresholds": [1 - t, t],
              "test": test, "C": c, "training_images": int(len(tr)), "mean": MEAN, "std": STD,
              "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "dermoscopy.json").write_text(json.dumps(result, indent=1))
    print("wrote dermoscopy", flush=True)


if __name__ == "__main__":
    main()
