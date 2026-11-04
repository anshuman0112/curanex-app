"""Adapter smoke test for mlmed_torchxrayvision, run inside the worker image (its packages aren't in the API env):

  docker run --rm --network none -e PYTHONPATH=/app -v <backend>/workers:/app/workers:ro -v <backend>/tools:/tools:ro \
      -v <backend>/tests/golden:/golden:ro -v <repo>/tests:/samples:ro backend-worker-torch python /tools/smoke_xrv.py
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from workers.torch.adapters.mlmed_torchxrayvision import Adapter, preprocess


def b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def refused(adapter: Adapter, img: Image.Image) -> bool:
    try:
        adapter.predict({"image_b64": b64(img)})
    except ValueError:
        return True
    return False


def main() -> None:
    adapter = Adapter()
    adapter.load()
    for case in json.loads(Path("/golden/mlmed_torchxrayvision.json").read_text()):
        scores, _ = adapter.scores(preprocess(Image.open(Path("/samples") / case["image"])))
        for label, expected in case["scores"].items():
            assert abs(float(scores[0, adapter.labels.index(label)]) - expected) < 1e-3, (case["image"], label)
    print("golden cases match")

    sample = Image.open(Path("/samples/00000001_000.png"))
    pred = adapter.predict({"image_b64": b64(sample)})
    assert "Abnormal X-ray" in pred.probabilities and "Lung opacity signs" in pred.probabilities and pred.explanation["group"] == "adult"
    child = adapter.predict({"image_b64": b64(sample), "age": 4})
    assert "Pneumonia signs" in child.probabilities and "Abnormal X-ray" not in child.probabilities
    assert "No clear" in child.label or "Possible" in child.label
    assert base64.b64decode(pred.explanation["heatmap_png_b64"])[:4] == b"\x89PNG"
    print("prediction + heatmap ok:", pred.label)

    colour = Image.merge("RGB", [Image.new("L", (400, 400), 220), Image.new("L", (400, 400), 90), Image.new("L", (400, 400), 30)])
    assert refused(adapter, colour), "colour photo must be refused"
    assert refused(adapter, Image.fromarray(np.zeros((80, 80), dtype=np.uint8))), "tiny image must be refused"
    assert refused(adapter, sample.resize((900, 200))), "wrong shape must be refused"
    assert adapter.validate({}) == ["image_b64: missing"]
    print("bad inputs refused")


if __name__ == "__main__":
    main()
