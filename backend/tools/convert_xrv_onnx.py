"""WORKFLOW.md A4-A5 for mlmed/torchxrayvision: load the pickled weights and export ONNX. Sandbox only:

  docker run --rm --network none -v <models>:/weights -v <backend>/workers:/app/workers:ro \
      -v <backend>/tools:/tools:ro -v <repo>/tests:/samples:ro -v <backend>/tests/golden:/golden \
      backend-worker-torch python /tools/convert_xrv_onnx.py

Checks the weights' SHA-256, exports one graph with two outputs (logits; one activation map per finding, so the
worker can draw heatmaps without gradients or pickles), compares ONNX with the library on 20 inputs (max abs diff
< 1e-4) and writes model_meta.json plus a golden case.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F  # noqa: N812
import torchxrayvision as xrv
from PIL import Image

from workers.torch.adapters.mlmed_torchxrayvision import op_norm, preprocess

DIR = Path("/weights")
WEIGHTS = DIR / "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
WEIGHTS_SHA256 = "56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"
ONNX = DIR / "model.onnx"


class Export(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.classifier = model.classifier

    def forward(self, x):
        maps = F.relu(self.features(x))
        logits = self.classifier(maps.mean(dim=(2, 3)))
        cams = torch.einsum("kc,bchw->bkhw", self.classifier.weight, maps)
        return logits, cams


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert sha256(WEIGHTS) == WEIGHTS_SHA256, "weights SHA-256 mismatch"
    model = xrv.models.DenseNet(weights="densenet121-res224-all", cache_dir=str(DIR)).eval()
    torch.onnx.export(Export(model).eval(), (torch.zeros(1, 1, 224, 224),), str(ONNX), opset_version=17, dynamo=False,
                      input_names=["image"], output_names=["logits", "cams"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "cams": {0: "batch"}})

    labels = list(model.pathologies)
    thresholds = model.op_threshs.numpy().tolist()
    session = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    def onnx_scores(batch: np.ndarray) -> np.ndarray:
        logits, _ = session.run(None, {"image": batch})
        return op_norm(1 / (1 + np.exp(-logits)), np.array(thresholds))

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        x = rng.uniform(-1024, 1024, (1, 1, 224, 224)).astype(np.float32)
        with torch.no_grad():
            lib = model(torch.from_numpy(x)).numpy()
        worst = max(worst, float(np.nanmax(np.abs(lib - onnx_scores(x)))))
    samples = sorted(Path("/samples").glob("*.png")) + sorted(Path("/samples").glob("*.jpg"))
    goldens = []
    for path in samples[:4]:
        x = preprocess(Image.open(path))
        with torch.no_grad():
            lib = model(torch.from_numpy(x)).numpy()
        ours = onnx_scores(x)
        worst = max(worst, float(np.nanmax(np.abs(lib - ours))))
        goldens.append({"image": path.name, "scores": {k: round(float(v), 5) for k, v in zip(labels, ours[0], strict=True)}})
    print(f"max abs diff library vs ONNX: {worst:.2e}")
    assert worst < 1e-4, "ONNX output differs from the library"

    (DIR / "model_meta.json").write_text(json.dumps({
        "labels": labels, "op_threshs": thresholds, "weights_sha256": WEIGHTS_SHA256,
        "onnx_sha256": sha256(ONNX), "max_abs_diff": worst, "torchxrayvision_version": xrv.__version__,
    }, indent=1))
    Path("/golden/mlmed_torchxrayvision.json").write_text(json.dumps(goldens, indent=1))
    print(f"wrote {ONNX} ({ONNX.stat().st_size / 1e6:.1f} MB), meta and {len(goldens)} golden cases")


if __name__ == "__main__":
    main()
