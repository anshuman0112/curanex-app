"""WORKFLOW.md A4-A5 for OmerKurtulus/skin-disease-classification. Sandbox only (the .keras file has Python Lambda layers):

  docker run --rm --network none -v <vendor>/omerkurtulus_skin_disease_classification:/repo:ro \
      -v <models>/omerkurtulus_skin_disease_classification:/out -v <backend>/tools:/tools:ro backend-worker-tf python /tools/convert_skin.py
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

REPO = Path("/repo/models/Model_10_SalvationNet_best.keras")
OUT = Path("/out")
CLASSES = ["acne", "hyperpigmentation", "Nail_psoriasis", "SJS-TEN", "Vitiligo"]


def main() -> None:
    print("tf", tf.__version__, "keras", tf.keras.__version__, flush=True)
    model = tf.keras.models.load_model(REPO, safe_mode=False, compile=False)
    print("loaded", model.name, "inputs", [str(i.shape) for i in model.inputs], "outputs", [str(o.shape) for o in model.outputs], flush=True)
    x = np.random.default_rng(0).uniform(0, 255, (4, 224, 224, 3)).astype(np.float32)
    original = model.predict(x, verbose=0)
    print("sample output", np.round(original[0], 3), flush=True)

    fn = tf.function(lambda inp: model(inp, training=False))
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image"),)
    onnx_model, _ = tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=str(OUT / "model.onnx"))
    session = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
    ours = session.run(None, {session.get_inputs()[0].name: x})[0]
    worst = float(np.abs(ours - original).max())
    print(f"max abs diff ONNX vs Keras: {worst:.2e}", flush=True)
    assert worst < 1e-3, "ONNX output differs"
    (OUT / "model_meta.json").write_text(json.dumps({
        "classes": CLASSES, "image_size": 224, "source_sha256": hashlib.sha256(REPO.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256((OUT / "model.onnx").read_bytes()).hexdigest(), "max_abs_diff": worst}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
