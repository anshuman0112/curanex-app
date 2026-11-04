"""WORKFLOW.md A4-A5 for AndreyGermanov/yolov8_caries_detector. Sandbox only (best.pt is a pickle):

  docker run --rm --network none -v <vendor>/andreygermanov_yolov8_caries_detector:/repo:ro \
      -v <models>/andreygermanov_yolov8_caries_detector:/out -v <backend>/tools:/tools:ro backend-worker-torch python /tools/convert_caries.py
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

OUT = Path("/out")


def main() -> None:
    src = Path("/repo/best.pt")
    work = Path("/tmp/best.pt")
    shutil.copy(src, work)
    model = YOLO(str(work))
    print("classes", model.names, flush=True)
    onnx_path = Path(model.export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False))
    shutil.copy(onnx_path, OUT / "model.onnx")
    s = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
    print("input", s.get_inputs()[0].name, s.get_inputs()[0].shape, "output", s.get_outputs()[0].shape, flush=True)

    x = np.random.default_rng(0).uniform(0, 1, (1, 3, 640, 640)).astype(np.float32)
    ours = s.run(None, {s.get_inputs()[0].name: x})[0]
    import torch
    with torch.no_grad():
        lib = model.model.float().eval()(torch.from_numpy(x))[0].numpy()
    worst = float(np.abs(ours - lib).max())
    print(f"max abs diff ONNX vs PyTorch: {worst:.2e}", flush=True)
    assert worst < 1e-2, "ONNX output differs"
    (OUT / "model_meta.json").write_text(json.dumps({
        "classes": {str(k): v for k, v in model.names.items()}, "imgsz": 640, "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256((OUT / "model.onnx").read_bytes()).hexdigest(), "max_abs_diff": worst}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
