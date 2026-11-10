"""Cancer image models trained on one dataset and tested once on a different one. Trained by the Curanex team.
From backend/: python training/train_outside_head.py <config>

Same recipe as train_folder_head.py (pretrained EfficientNet-B0 features + logistic layer, CPU only), but the test set is
a separate collection from another lab or country, so the result shows how the model does on images it has never seen
the like of. Rules fixed before testing:
- exact duplicate files removed; training data split 85/15 for validation (C and thresholds)
- each class threshold: highest with validation sensitivity >= 0.85
- outside test scored once; a class passes only with outside sensitivity >= 0.80 and AUC >= 0.80
- exported always (for the record); switched on only if the finding class passes

Configs:
- crc: colorectal cancer in H&E pathology tiles. Train Kather 2016 (Mannheim, 5,000 tiles; "complex stroma" left out as
  mixed), test CRC-VAL-HE-7K (Kather 2018, NCT Biobank, 7,180 tiles from 50 other patients). Tumour against everything else.
- cervix: abnormal cervical cells in Pap smears. Train SIPaKMeD (Greece, cropped cells), test Herlev (Denmark).
  Abnormal = dyskeratotic or koilocytotic (SIPaKMeD); dysplasia or carcinoma in situ (Herlev).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from training.train_folder_head import IMAGE, MEAN, OUT, STD, prep  # noqa: E402

EVAL = BACKEND / "data" / "eval"
MAX_PER_CLASS = 2500


def crc_train(p: Path) -> str | None:
    folder = p.parent.name.upper()
    return None if "COMPLEX" in folder else "tumour" if "TUMOR" in folder else "no_tumour"


def crc_test(p: Path) -> str | None:
    return "tumour" if p.parent.name.upper() == "TUM" else "no_tumour"


def cervix_train(p: Path) -> str | None:
    if "CROPPED" not in str(p).upper():
        return None
    s = str(p).lower()
    return "abnormal" if ("dyskeratotic" in s or "koilocytotic" in s) else "normal"


def cervix_test(p: Path) -> str | None:
    if "-d" in p.stem.lower():  # Herlev stores a segmentation image beside each cell
        return None
    s = p.parent.name.lower()
    return "abnormal" if ("dysplas" in s or "carcinoma" in s) else "normal"


CONFIGS = {
    "crc": {"title": "Colorectal cancer (pathology slide tiles)", "train": EVAL / "crc" / "kather2016", "test": EVAL / "crc" / "crcval",
            "train_label": crc_train, "test_label": crc_test, "train_name": "Kather 2016 (Mannheim, Germany)",
            "test_name": "CRC-VAL-HE-7K (NCT Biobank, Heidelberg; 50 other patients)"},
    "cervix": {"title": "Abnormal cervical cells (Pap smear)", "train": EVAL / "cervix" / "sipakmed", "test": EVAL / "cervix" / "herlev",
               "train_label": cervix_train, "test_label": cervix_test, "train_name": "SIPaKMeD (Ioannina, Greece)",
               "test_name": "Herlev (Copenhagen, Denmark)"},
}


def gather(root: Path, label_of, seen: set) -> list[tuple[Path, str]]:
    rows, counts, per_folder = [], {}, {}
    for f in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE):
        label = label_of(f)
        # capped per source folder, so every tissue or cell type stays represented
        if label is None or per_folder.get(f.parent, 0) >= MAX_PER_CLASS:
            continue
        per_folder[f.parent] = per_folder.get(f.parent, 0) + 1
        h = hashlib.md5(f.read_bytes()).hexdigest()  # noqa: S324 finds duplicates, not security
        if h in seen:
            continue
        seen.add(h)
        rows.append((f, label))
        counts[label] = counts.get(label, 0) + 1
    print(root.name, counts, flush=True)
    return rows


def features(net, rows, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["keep"]
    X, keep = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 24):
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
            if i % 960 == 0:
                print("features", cache.stem, i, flush=True)
    X, keep = np.concatenate(X), np.array(keep)
    np.savez(cache, X=X, keep=keep)
    return X, keep


def main(name: str) -> None:
    cfg = CONFIGS[name]
    torch.set_num_threads(3)
    seen: set = set()
    train_rows = gather(cfg["train"], cfg["train_label"], seen)
    test_rows = gather(cfg["test"], cfg["test_label"], seen)
    classes = sorted({r[1] for r in train_rows})
    net = nn.Sequential(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features, nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    X, keep = features(net, train_rows, EVAL / f"features_{name}_train.npz")
    XT, keept = features(net, test_rows, EVAL / f"features_{name}_outside.npz")
    y = np.array([classes.index(train_rows[k][1]) for k in keep])
    yt = np.array([classes.index(test_rows[k][1]) for k in keept])
    tr, va = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X[tr])
    best = None
    for c in (0.001, 0.01, 0.1):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(scaler.transform(X[tr]), y[tr])
        a = roc_auc_score(y[va], m.predict_proba(scaler.transform(X[va]))[:, 1])
        print(f"C={c} val AUC {a:.4f}", flush=True)
        if best is None or a > best[0]:
            best = (a, c, m)
    _, c, model = best
    pv, pt = model.predict_proba(scaler.transform(X[va])), model.predict_proba(scaler.transform(XT))
    per, passed, thresholds = {}, [], []
    for i, label in enumerate(classes):
        pos = pv[y[va] == i, i]
        t = float(max(v for v in np.unique(pos) if (pos >= v).mean() >= 0.85))
        thresholds.append(t)
        truth, s = yt == i, pt[:, i]
        sens, spec, auc = float((s[truth] >= t).mean()), float((s[~truth] < t).mean()), float(roc_auc_score(truth, s))
        per[label] = {"n": int(truth.sum()), "sensitivity": round(sens, 3), "specificity": round(spec, 3), "auc": round(auc, 3)}
        if sens >= 0.80 and auc >= 0.80:
            passed.append(label)
    result = {"id": name, "title": cfg["title"], "labels": classes, "images": int(len(y)), "outside_images": int(len(yt)), "c": c,
              "accuracy": round(float(accuracy_score(yt, pt.argmax(1))), 3), "per_class": per, "passed": passed, "thresholds": thresholds,
              "mean": MEAN, "std": STD, "trained_on": cfg["train_name"], "outside_test": cfg["test_name"]}
    print(json.dumps({k: result[k] for k in ("id", "images", "outside_images", "accuracy", "passed")}), flush=True)
    for label, v in per.items():
        print(f"  outside {label}: {v}", flush=True)

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))
            self.register_buffer("coef", torch.tensor(np.vstack([-model.coef_, model.coef_]), dtype=torch.float32))
            self.register_buffer("icpt", torch.tensor(np.array([-model.intercept_[0], model.intercept_[0]]), dtype=torch.float32))

        def forward(self, x):
            return (((self.net(x) - self.mean) / self.scale) @ self.coef.T + self.icpt).softmax(1)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    torch.onnx.export(Full().eval(), (torch.from_numpy(prep(train_rows[keep[0]][0])[None]),), str(path), input_names=["image"], output_names=["probs"],
                      opset_version=17, dynamo=False, dynamic_axes={"image": {0: "n"}})
    result["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    print("wrote", name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
