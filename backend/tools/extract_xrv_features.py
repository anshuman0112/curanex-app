"""Cache the X-ray model's deep features + scores for Kermany train/test and the adult set. Worker image, no network:

  docker run --rm --network none -e PYTHONPATH=/app -v <backend>/data/eval:/data -v <backend>/workers:/app/workers:ro \
      -v <backend>/tools:/tools:ro backend-worker-torch python /tools/extract_xrv_features.py

Adult set (ieee8023/covid-chestxray-dataset): frontal X-rays of adults (age >= 18 when given), one image per patient;
any "Pneumonia" finding = positive, "No Finding" = negative.
"""
import csv
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from workers.torch.adapters.mlmed_torchxrayvision import Adapter, preprocess

DATA = Path("/data")
adapter = None


def init() -> None:
    global adapter
    adapter = Adapter()
    adapter.load()


def one(path: str):
    try:
        x = preprocess(Image.open(path))
    except Exception:  # unreadable or unsupported file in a public dataset
        return None
    logits, feats = adapter.session.run(["logits", "features"], {"image": x})
    scores, _ = adapter.scores(x)
    return feats[0], scores[0]


def kermany(split: str):
    items = []
    for folder, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
        for p in sorted((DATA / "chest_xray" / split / folder).glob("*.jpeg")):
            m = re.match(r"(person\d+|NORMAL2-IM-\d+|IM-\d+)", p.name)
            items.append((p, label, m.group(1) if m else p.stem))
    return items


def adults():
    root = DATA / "covid-chestxray-dataset"
    seen, items = set(), []
    for r in csv.DictReader((root / "metadata.csv").open(encoding="utf-8")):
        finding = r["finding"]
        label = 1 if "Pneumonia" in finding else 0 if finding == "No Finding" else None
        age = float(r["age"]) if r["age"].replace(".", "").isdigit() else None
        path = root / "images" / r["filename"]
        if (label is None or r["modality"] != "X-ray" or not r["view"].startswith(("PA", "AP")) or
                (age is not None and age < 18) or r["patientid"] in seen or not path.exists()):
            continue
        seen.add(r["patientid"])
        items.append((path, label, r["patientid"]))
    return items


def save(name: str, items: list) -> None:
    out = DATA / f"features_{name}.npz"
    with Pool(4, initializer=init) as pool:
        rows = pool.map(one, [str(p) for p, _, _ in items], chunksize=16)
    keep = [i for i, r in enumerate(rows) if r is not None]
    np.savez(out, X=np.stack([rows[i][0] for i in keep]), scores=np.stack([rows[i][1] for i in keep]),
             y=np.array([items[i][1] for i in keep]), group=np.array([str(items[i][2]) for i in keep]))
    print(f"{name}: {len(keep)} of {len(items)} images, positives {sum(items[i][1] for i in keep)}", flush=True)


if __name__ == "__main__":
    save("adult", adults())
    save("test", kermany("test"))
    save("train", kermany("train"))
