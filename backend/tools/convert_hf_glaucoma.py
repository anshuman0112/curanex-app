"""Exports pamixsun/swinv2_tiny_for_glaucoma_classification (Apache-2.0, safetensors only) to ONNX. Sandbox only:

  docker run --rm --network none -m 2800m -v <vendor>/pamixsun_glaucoma:/repo:ro -v <models>/pamixsun_glaucoma:/out \\
      -v <backend>/tools:/tools:ro backend-worker-torch python /tools/convert_hf_glaucoma.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import Swinv2ForImageClassification

REPO = Path("/repo")
OUT = Path("/out")


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits


def main() -> None:
    model = Swinv2ForImageClassification.from_pretrained(REPO, use_safetensors=True, local_files_only=True).eval()
    wrapped = Logits(model).eval()
    x = torch.from_numpy(np.random.default_rng(0).normal(0, 1, (1, 3, 512, 512)).astype(np.float32))
    path = OUT / "model.onnx"
    torch.onnx.export(wrapped, (x,), str(path), input_names=["image"], output_names=["logits"], opset_version=17, do_constant_folding=True, dynamo=False)
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        worst = float(np.abs(s.run(None, {"image": x.numpy()})[0] - wrapped(x).numpy()).max())
    print("max abs diff", worst, flush=True)
    assert worst < 1e-2
    (OUT / "convert.json").write_text(json.dumps({"source_sha256": hashlib.sha256((REPO / "model.safetensors").read_bytes()).hexdigest(),
                                                  "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "max_abs_diff": worst}, indent=1))


if __name__ == "__main__":
    main()
