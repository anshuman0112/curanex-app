"""Breast ultrasound, malignant or not, trained by the Curanex team. From backend/: python training/train_breast.py

BUSI alone missed about a third of cancers (sensitivity 0.645). Rules fixed first: train on BUS-BRA split by patient (Case), with
threshold from BUS-BRA validation, sensitivity >= 0.90 because a missed cancer is the worse error. BUSI is the outside test,
scored once, removing masks. Exported only if outside sensitivity and AUC reach 0.80.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from training.train_skin import MEAN, STD, backbone, prep  # noqa: E402

BRA = BACKEND / "data" / "eval" / "busbra" / "x" / "BUSBRA" / "BUSBRA"
BUSI = BACKEND / "data" / "eval" / "busi" / "x" / "Dataset_BUSI_with_GT"
FEATS = BACKEND / "data" / "eval" / "features_breast.npz"
OUT = BACKEND / "workers" / "torch" / "models" / "multi"


def report(y, s, t):
    tn, fp, fn, tp = confusion_matrix(y, s >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, s)), 3)}


def main() -> None:
    torch.set_num_threads(4)
    with (BRA / "bus_data.csv").open(encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    rows = [(BRA / "Images" / f"{r['ID']}.png", int(r["Pathology"] == "malignant"), r["Case"], "bra") for r in table if (BRA / "Images" / f"{r['ID']}.png").exists()]
    for c in ("benign", "malignant"):
        rows += [(f, int(c == "malignant"), f.stem, "busi") for f in sorted((BUSI / c).glob("*.png")) if "_mask" not in f.name]
    print(len(rows), flush=True)
    net = backbone()
    if FEATS.exists():
        X = np.load(FEATS)["X"]
    else:
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 24):
                out.append(net(torch.from_numpy(np.stack([prep(r[0].read_bytes()) for r in rows[i:i + 24]]))).numpy())
                if i % 480 == 0:
                    print("features", i, flush=True)
        X = np.concatenate(out)
        np.savez(FEATS, X=X)
    y = np.array([r[1] for r in rows])
    src = np.array([r[3] for r in rows])
    group = np.array([r[2] for r in rows])
    bra = np.where(src == "bra")[0]
    a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(bra, y[bra], group[bra]))
    tr, va = bra[a], bra[b]
    sc = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        s = roc_auc_score(y[va], m.predict_proba(sc.transform(X[va]))[:, 1])
        print(f"C={c} BUS-BRA validation AUC {s:.4f}", flush=True)
        if best is None or s > best[0]:
            best = (s, c, m)
    _, c, model = best
    score = lambda Z: model.predict_proba(sc.transform(Z))[:, 1]  # noqa: E731
    sv = score(X[va])
    t = float(max(v for v in np.unique(sv) if (sv[y[va] == 1] >= v).mean() >= 0.90))
    busi = src == "busi"
    result = {"threshold": t, "C": c, "busbra_validation": report(y[va], sv, t), "outside_busi": report(y[busi], score(X[busi]), t), "mean": MEAN, "std": STD}
    print(json.dumps({k: result[k] for k in ("busbra_validation", "outside_busi")}), flush=True)
    r = result["outside_busi"]
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
    torch.onnx.export(Full().eval(), (sample,), str(path), input_names=["image"], output_names=["score"], opset_version=17, dynamo=False, dynamic_axes={"image": {0: "n"}})
    result["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "breast.json").write_text(json.dumps(result, indent=1))
    print("wrote breast", flush=True)


if __name__ == "__main__":
    main()
