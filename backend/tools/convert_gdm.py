"""WORKFLOW.md A4-A5 for srishti-chordia/Gestational-Diabetes-Prediction-using-DL: .h5 to ONNX. Sandbox only:

  docker run --rm --network none -v <vendor>/srishti_chordia_gestational_diabetes_prediction_using_dl:/repo:ro       -v <models>/srishti_chordia_gestational_diabetes_prediction_using_dl:/out -v <backend>/tools:/tools:ro backend-worker-tf python /tools/convert_gdm.py
Then tools/evaluate_gdm.py in the tabular image.
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


def main() -> None:
    src = REPO / "gdm_model.h5"
    model = tf.keras.models.load_model(src, compile=False)
    print("loaded", [str(i.shape) for i in model.inputs], [str(o.shape) for o in model.outputs], flush=True)
    fn = tf.function(lambda inp: model(inp, training=False))
    spec = (tf.TensorSpec((None, 25), tf.float32, name="x"),)  # the saved model takes all 25 columns, not the 18 in the repo app
    tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=str(OUT / "model.onnx"))
    session = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(0, 1, (50, 25)).astype(np.float32)
    worst = float(np.abs(session.run(None, {session.get_inputs()[0].name: x})[0] - model.predict(x, verbose=0)).max())
    print(f"max abs diff ONNX vs Keras: {worst:.2e}", flush=True)
    assert worst < 1e-3
    (OUT / "convert.json").write_text(json.dumps({"source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "max_abs_diff": worst}))


if __name__ == "__main__":
    main()
