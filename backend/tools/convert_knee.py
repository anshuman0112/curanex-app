"""WORKFLOW.md A4-A8 for Ghifar-Khder/knee-osteoarthritis-classification. Sandbox only:

  docker run --rm --network none -v <vendor>/ghifar_khder_knee_osteoarthritis_classification:/repo:ro \
      -v <models>/ghifar_khder_knee_osteoarthritis_classification:/out -v <backend>/tools:/tools:ro backend-worker-tf python /tools/convert_knee.py

Measured on the repo's held-out Test-data (one folder per KL grade). Primary metric, fixed before running: KL grade 2 or
higher vs 0-1, using the model's summed probability of grades 2-4 at 0.5.
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
from PIL import Image  # noqa: E402

REPO = Path("/repo")
OUT = Path("/out")
SIZE = 224


def auc(y, s):
    order = s.argsort()
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        r[s == v] = r[s == v].mean()
    pos, neg = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> None:
    src = REPO / "final-model.keras"
    model = tf.keras.models.load_model(src, compile=False)
    print("loaded", model.name, [str(i.shape) for i in model.inputs], [str(o.shape) for o in model.outputs], flush=True)
    fn = tf.function(lambda inp: model(inp, training=False))
    spec = (tf.TensorSpec((None, SIZE, SIZE, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=str(OUT / "model.onnx"))
    session = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def run(batch):
        out = session.run(None, {name: batch})
        return out[0]

    x = np.random.default_rng(0).uniform(0, 1, (2, SIZE, SIZE, 3)).astype(np.float32)
    lib = model.predict(x, verbose=0)
    lib = lib[0] if isinstance(lib, list) else lib
    worst = float(np.abs(run(x) - lib).max())
    print(f"max abs diff ONNX vs Keras: {worst:.2e}", flush=True)
    assert worst < 1e-3, "ONNX output differs"

    probs, grades = [], []
    for grade in range(5):
        for path in sorted((REPO / "Test-data" / str(grade)).glob("*")):
            img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
            probs.append(run(np.asarray(img, dtype=np.float32)[np.newaxis] / 255.0)[0])
            grades.append(grade)
    p, g = np.array(probs), np.array(grades)
    y = (g >= 2).astype(int)
    s = p[:, 2:].sum(axis=1)
    f = s >= 0.5
    tp, fn_, tn, fp = int((f & (y == 1)).sum()), int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum()), int((f & (y == 0)).sum())
    metrics = {"n": len(y), "positives": int(y.sum()), "accuracy": round((tp + tn) / len(y), 3), "sensitivity": round(tp / (tp + fn_), 3),
               "specificity": round(tn / (tn + fp), 3), "auc": round(auc(y, s), 3), "confusion": {"tp": tp, "fn": fn_, "tn": tn, "fp": fp},
               "five_grade_accuracy": round(float((p.argmax(axis=1) == g).mean()), 3), "per_grade_n": np.bincount(g).tolist()}
    (OUT / "model_meta.json").write_text(json.dumps({
        "image_size": SIZE, "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256((OUT / "model.onnx").read_bytes()).hexdigest(), "max_abs_diff": worst, "metrics": metrics}, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
