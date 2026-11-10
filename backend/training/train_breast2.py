"""Breast ultrasound, malignant or not: second attempt, trained by the Curanex team. From backend/: python training/train_breast2.py

The first attempt (train_breast.py) trained on BUS-BRA (Brazil) and reached AUC 0.788 on BUSI (Egypt): failed.
Rules fixed before looking at the new test set:
- train on BUS-BRA and BUSI together (more variety of machines and hospitals); validation = 25% of BUS-BRA cases by patient
- same EfficientNet-B0 features and logistic layer; C chosen on validation AUC
- threshold: highest with validation sensitivity >= 0.90 (a missed cancer is the worse error)
- outside test, scored once: BrEaST (Poland, 2019-2022, 256 patients), benign against malignant, masks not used
- exported only if outside sensitivity and AUC both reach 0.80
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from training.train_breast import BRA, BUSI, FEATS, OUT, report  # noqa: E402
from training.train_skin import MEAN, STD, backbone, prep  # noqa: E402

BREAST = BACKEND / "data" / "eval" / "breast_test"
BREAST_FEATS = BACKEND / "data" / "eval" / "features_breast_test.npz"


def training_rows():
    import csv

    with (BRA / "bus_data.csv").open(encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    rows = [(BRA / "Images" / f"{r['ID']}.png", int(r["Pathology"] == "malignant"), r["Case"], "bra") for r in table if (BRA / "Images" / f"{r['ID']}.png").exists()]
    for c in ("benign", "malignant"):
        rows += [(f, int(c == "malignant"), f.stem, "busi") for f in sorted((BUSI / c).glob("*.png")) if "_mask" not in f.name]
    return rows


def main() -> None:
    torch.set_num_threads(4)
    rows = training_rows()
    X = np.load(FEATS)["X"]  # features from train_breast.py, same row order
    assert len(X) == len(rows)
    clinical = pd.read_excel(BREAST / "clinical.xlsx")
    clinical = clinical[clinical["Classification"].isin(["benign", "malignant"])]
    test_files = [BREAST / "BrEaST-Lesions_USG-images_and_masks" / f for f in clinical["Image_filename"]]
    net = backbone()
    if BREAST_FEATS.exists():
        XT = np.load(BREAST_FEATS)["X"]
    else:
        with torch.no_grad():
            XT = np.concatenate([net(torch.from_numpy(np.stack([prep(p.read_bytes()) for p in test_files[i:i + 24]]))).numpy()
                                 for i in range(0, len(test_files), 24)])
        np.savez(BREAST_FEATS, X=XT)
    yt = (clinical["Classification"] == "malignant").to_numpy(int)

    y = np.array([r[1] for r in rows])
    src = np.array([r[3] for r in rows])
    group = np.array([r[2] for r in rows])
    bra = np.where(src == "bra")[0]
    a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(bra, y[bra], group[bra]))
    va = bra[b]
    tr = np.setdiff1d(np.arange(len(y)), va)
    sc = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        s = roc_auc_score(y[va], m.predict_proba(sc.transform(X[va]))[:, 1])
        print(f"C={c} validation AUC {s:.4f}", flush=True)
        if best is None or s > best[0]:
            best = (s, c, m)
    _, c, model = best
    score = lambda Z: model.predict_proba(sc.transform(Z))[:, 1]  # noqa: E731
    sv = score(X[va])
    t = float(max(v for v in np.unique(sv) if (sv[y[va] == 1] >= v).mean() >= 0.90))
    result = {"threshold": t, "C": c, "validation": report(y[va], sv, t), "outside_breast_poland": report(yt, score(XT), t), "mean": MEAN, "std": STD,
              "trained_on": f"{len(tr)} ultrasound images from BUS-BRA (Brazil) and BUSI (Egypt)"}
    print(json.dumps({k: result[k] for k in ("validation", "outside_breast_poland")}), flush=True)
    r = result["outside_breast_poland"]
    if r["sensitivity"] < 0.80 or r["auc"] < 0.80:
        print("FAILED: not exported", flush=True)
        return

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(sc.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(sc.scale_, dtype=torch.float32))
            self.register_buffer("coef", torch.tensor(model.coef_[0], dtype=torch.float32))
            self.intercept = float(model.intercept_[0])

        def forward(self, x):
            return torch.sigmoid((((self.net(x) - self.mean) / self.scale) * self.coef).sum(1) + self.intercept)

    path = OUT / "breast.onnx"
    sample = torch.from_numpy(prep(rows[0][0].read_bytes())[None])
    torch.onnx.export(Full().eval(), (sample,), str(path), input_names=["image"], output_names=["score"], opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "n"}})
    result["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "breast.json").write_text(json.dumps(result, indent=1))
    print("wrote breast", flush=True)


if __name__ == "__main__":
    main()
