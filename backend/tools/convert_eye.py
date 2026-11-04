"""WORKFLOW.md A4-A5 for khaled-ghanem/Ocular-Disease-Classification: one .h5 to ONNX per run. Sandbox only:

  docker run --rm --network none -m 3400m -e TF_USE_LEGACY_KERAS=1 -v <vendor>/khaled_ghanem_ocular_disease_classification:/repo:ro \
      -v <models>/khaled_ghanem_ocular_disease_classification:/out -v <backend>/tools:/tools:ro backend-worker-tf python /tools/convert_eye.py cataract

tf2onnx's graph optimizer is what exhausts memory on these models, so it is skipped; onnxruntime optimizes at load anyway.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import tensorflow as tf  # noqa: E402
import tf2onnx  # noqa: E402
from tf2onnx import optimizer  # noqa: E402

REPO = Path("/repo")
OUT = Path("/out")
MODELS = {"cataract": "cataract/saved_model/ResNet50-Cataract_classification-98.98.h5", "glaucoma": "glaucoma/saved_model/VGG19-Glaucoma_classification-98.91.h5"}
optimizer.optimize_graph = lambda graph, *_, **__: graph


def main() -> None:
    name = sys.argv[1]
    src = REPO / MODELS[name]
    local = Path(shutil.copy(src, "/tmp/model.h5"))  # the Windows bind mount makes h5 loading take minutes
    model = tf.keras.models.load_model(local, compile=False)
    print(name, "loaded", [str(i.shape) for i in model.inputs], [str(o.shape) for o in model.outputs], flush=True)
    fn = tf.function(lambda inp, m=model: m(inp, training=False))
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image"),)
    path = OUT / f"{name}.onnx"
    tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=str(path))
    print(name, "exported", path.stat().st_size, flush=True)
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)
    worst = float(np.abs(s.run(None, {s.get_inputs()[0].name: x})[0] - model.predict(x, verbose=0)).max())
    print(f"{name} max abs diff ONNX vs Keras: {worst:.2e}", flush=True)
    assert worst < 1e-3
    meta_path = OUT / "convert.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta[name] = {"source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "max_abs_diff": worst}
    meta_path.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
