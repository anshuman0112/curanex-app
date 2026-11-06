"""Bone age, second attempt. From backend/: python training/train_boneage2.py

The ridge layer missed the bar at 13.4 months. Rules fixed first: keep the 6,000 images already used for training and validation,
add a small neural layer, and score on 2,000 images from the same file that the first attempt never touched, once.
Pass is a mean error of 12 months or less. Exported only on a pass.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from training.train_boneage import FEATS, LIMIT, OUT, ZIP, prep  # noqa: E402
from training.train_skin import MEAN, STD, backbone  # noqa: E402

FRESH = 2000
EXTRA = BACKEND / "data" / "eval" / "features_boneage_fresh.npz"


class Head(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.body(x).squeeze(1)


def main() -> None:
    torch.set_num_threads(3)
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    table = next(n for n in names if n.endswith("boneage-training-dataset.csv"))
    files = {Path(n).stem: n for n in names if n.endswith(".png") and "training" in n}
    rows = [ln.split(",") for ln in z.read(table).decode().splitlines()[1:] if ln]
    rows = [r for r in rows if r[0] in files]
    np.random.default_rng(42).shuffle(rows)  # same order as the first attempt
    seen, fresh = rows[:LIMIT], rows[LIMIT:LIMIT + FRESH]
    net = backbone()
    if EXTRA.exists():
        Xf = np.load(EXTRA)["X"]
    else:
        out = []
        with torch.no_grad():
            for i in range(0, len(fresh), 24):
                out.append(net(torch.from_numpy(np.stack([prep(z.read(files[r[0]])) for r in fresh[i:i + 24]]))).numpy())
                if i % 480 == 0:
                    print("features", i, flush=True)
        Xf = np.concatenate(out)
        np.savez(EXTRA, X=Xf)
    male = lambda rs: np.array([[float(r[2].strip().lower() == "true")] for r in rs], dtype=np.float32)  # noqa: E731
    X = np.c_[np.load(FEATS)["X"], male(seen)].astype(np.float32)
    y = np.array([float(r[1]) for r in seen], dtype=np.float32)
    Xt = np.c_[Xf, male(fresh)].astype(np.float32)
    yt = np.array([float(r[1]) for r in fresh], dtype=np.float32)
    mean, scale = X.mean(0), X.std(0) + 1e-6
    tx, ty = torch.from_numpy((X - mean) / scale), torch.from_numpy(y)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    va, tr = idx[:900], idx[900:]
    best, state = 1e9, None
    torch.manual_seed(42)
    head = Head(X.shape[1])
    opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-3)
    for epoch in range(120):
        head.train()
        for b in torch.from_numpy(rng.permutation(tr)).split(128):
            opt.zero_grad()
            nn.functional.l1_loss(head(tx[b]), ty[b]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            e = float((head(tx[va]) - ty[va]).abs().mean())
        if e < best:
            best, state = e, {k: v.clone() for k, v in head.state_dict().items()}
        if epoch % 20 == 0:
            print(epoch, round(e, 2), flush=True)
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        p = head(torch.from_numpy((Xt - mean) / scale)).numpy()
    mae = float(np.abs(p - yt).mean())
    result = {"id": "boneage", "val_mae_months": round(best, 2), "test_mae_months": round(mae, 2), "n": len(yt), "passed": mae <= 12, "mean": MEAN, "std": STD}
    print(json.dumps(result), flush=True)
    if mae > 12:
        print("FAILED: not exported", flush=True)
        return

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.net, self.head = net, head
            self.register_buffer("m", torch.from_numpy(mean))
            self.register_buffer("s", torch.from_numpy(scale))

        def forward(self, x, is_male):
            return self.head((torch.cat([self.net(x), is_male], 1) - self.m) / self.s)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "boneage.onnx"
    sample = torch.from_numpy(prep(z.read(files[fresh[0][0]]))[None])
    torch.onnx.export(Full().eval(), (sample, torch.ones(1, 1)), str(path), input_names=["image", "male"], output_names=["months"], opset_version=17, dynamo=False)
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "boneage.json").write_text(json.dumps(result, indent=1))
    print("wrote boneage", flush=True)


if __name__ == "__main__":
    main()
