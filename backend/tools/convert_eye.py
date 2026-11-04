"""WORKFLOW.md A4-A5 for khaled-ghanem/Ocular-Disease-Classification: cataract and glaucoma .h5 to ONNX. Sandbox only:

  docker run --rm --network none -v <vendor>/khaled_ghanem_ocular_disease_classification:/repo:ro \
      -v <models>/khaled_ghanem_ocular_disease_classification:/out -v <backend>/tools:/tools:ro backend-worker-tf python /tools/convert_eye.py
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import tensorflow as tf  # noqa: E402
import tf2onnx  # noqa: E402

REPO = Path("/repo")
OUT = Path("/out")
MODELS = {"cataract": "cataract/saved_model/ResNet50-Cataract_classification-98.98.h5", "glaucoma": "glaucoma/saved_model/VGG19-Glaucoma_classification-98.91.h5"}


def main() -> None:
    meta = {}
    for name, rel in MODELS.items():
        src = REPO / rel
        model = tf.keras.models.load_model(src, compile=False)
        print(name, "loaded", [str(i.shape) for i in model.inputs], [str(o.shape) for o in model.outputs], flush=True)
        fn = tf.function(lambda inp, m=model: m(inp, training=False))
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image"),)
        path = OUT / f"{name}.onnx"
        tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=str(path))
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).uniform(0, 1, (2, 224, 224, 3)).astype(np.float32)
        worst = float(np.abs(s.run(None, {s.get_inputs()[0].name: x})[0] - model.predict(x, verbose=0)).max())
        print(f"{name} max abs diff ONNX vs Keras: {worst:.2e}", flush=True)
        assert worst < 1e-3
        meta[name] = {"source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "max_abs_diff": worst}
    (OUT / "convert.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
