"""Small CNN on one MedMNIST v2 dataset, trained by the Curanex team. Run from backend/: python training/train_medmnist.py dermamnist

Experimental only: these images are 28x28 pixels, so the models can't read a real patient's photo or scan.
Rules: official train/val/test splits; best epoch chosen on validation AUC; the official test split is scored once.
Nothing here loads downloaded weights: the model is built and trained from scratch, then exported to ONNX.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "eval" / "medmnist"
OUT = BACKEND / "workers" / "torch" / "models" / "medmnist"
TASKS = {
    "dermamnist": ("Skin lesion type", ["Actinic keratosis", "Basal cell carcinoma", "Benign keratosis", "Dermatofibroma", "Melanoma", "Melanocytic nevus", "Vascular lesion"]),
    "bloodmnist": ("Blood cell type", ["Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"]),
    "octmnist": ("Retinal OCT finding", ["Choroidal neovascularization", "Diabetic macular edema", "Drusen", "Normal"]),
    "breastmnist": ("Breast ultrasound", ["Malignant", "Normal or benign"]),
    "pathmnist": ("Colon tissue type", ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle", "Normal colon mucosa", "Cancer-associated stroma", "Colorectal adenocarcinoma"]),
    "retinamnist": ("Diabetic retinopathy grade", ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]),
    "pneumoniamnist": ("Child chest X-ray pneumonia", ["Normal", "Pneumonia"]),
    "organamnist": ("Abdominal CT organ, axial", ["Bladder", "Femur left", "Femur right", "Heart", "Kidney left", "Kidney right", "Liver", "Lung left", "Lung right", "Pancreas", "Spleen"]),
    "organcmnist": ("Abdominal CT organ, coronal", ["Bladder", "Femur left", "Femur right", "Heart", "Kidney left", "Kidney right", "Liver", "Lung left", "Lung right", "Pancreas", "Spleen"]),
    "organsmnist": ("Abdominal CT organ, sagittal", ["Bladder", "Femur left", "Femur right", "Heart", "Kidney left", "Kidney right", "Liver", "Lung left", "Lung right", "Pancreas", "Spleen"]),
    "tissuemnist": ("Kidney cortex cell type", ["Collecting duct", "Distal tubule", "Glomerular endothelial", "Interstitial endothelial", "Leukocyte", "Podocyte", "Proximal tubule", "Thick ascending limb"]),
}
EPOCHS = 14
BATCH = 128


class Net(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
        self.body = nn.Sequential(block(channels, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, classes))

    def forward(self, x):
        return self.body(x)


def tensor(a):
    a = a.astype(np.float32) / 255.0
    a = a[:, None] if a.ndim == 3 else a.transpose(0, 3, 1, 2)
    return torch.from_numpy((a - 0.5) / 0.5)


@torch.no_grad()
def probs(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 512]).softmax(1) for i in range(0, len(x), 512)]).numpy()


def score(y, p):
    a = roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")
    return {"n": len(y), "accuracy": round(float(accuracy_score(y, p.argmax(1))), 3), "auc": round(float(a), 3)}


def main(name: str) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    d = np.load(DATA / f"{name}.npz", allow_pickle=False)
    xtr, ytr = tensor(d["train_images"]), torch.from_numpy(d["train_labels"].ravel().astype(np.int64))
    xva, yva = tensor(d["val_images"]), d["val_labels"].ravel()
    xte, yte = tensor(d["test_images"]), d["test_labels"].ravel()
    title, labels = TASKS[name]
    model = Net(xtr.shape[1], len(labels))
    weight = torch.from_numpy((1 / np.maximum(np.bincount(ytr.numpy(), minlength=len(labels)), 1)).astype(np.float32))
    loss_fn = nn.CrossEntropyLoss(weight=weight / weight.sum() * len(labels))
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best, state = -1.0, None
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for b in torch.randperm(len(xtr)).split(BATCH):
            xb = xtr[b]
            if torch.rand(1) < 0.5:
                xb = xb.flip(3)
            opt.zero_grad()
            loss_fn(model(xb), ytr[b]).backward()
            opt.step()
        sched.step()
        a = score(yva, probs(model, xva))["auc"]
        print(f"{name} epoch {epoch + 1} val AUC {a} ({time.time() - start:.0f}s)", flush=True)
        if a > best:
            best, state = a, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    result = {"id": name, "title": title, "labels": labels, "channels": int(xtr.shape[1]), "val_auc": best, "test": score(yte, probs(model, xte)), "train_images": len(xtr)}
    print(json.dumps({k: result[k] for k in ("id", "val_auc", "test")}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    torch.onnx.export(model.eval(), (xte[:1],), str(path), input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        diff = float(np.abs(s.run(None, {"image": xte[:1].numpy()})[0] - model(xte[:1]).numpy()).max())
    assert diff < 1e-3
    result["onnx_diff"] = diff
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    print("wrote", name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
