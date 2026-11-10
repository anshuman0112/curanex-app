"""Fetches a balanced outside test set for the dermoscopy model from the public ISIC archive: BCN20000 (Hospital Clinic
Barcelona), independent of the HAM10000 training data. One image per lesion. Run from backend/."""

import json
import random
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "isic_bcn"
PER_CLASS = 750


def label(meta: dict) -> int | None:
    c = meta.get("clinical", {})
    d1, d3 = c.get("diagnosis_1", ""), (c.get("diagnosis_3") or "").lower()
    if "actinic keratosis" in d3 or d1 == "Malignant":
        return 1
    if d1 == "Benign":
        return 0
    return None  # indeterminate


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / "index.json"
    if index.exists():
        rows = json.loads(index.read_text())
    else:
        rows, url = [], "https://api.isic-archive.com/api/v2/images/search/?collections=249&limit=100"
        with httpx.Client(timeout=60) as c:
            while url:
                r = c.get(url)
                if r.status_code != 200:
                    time.sleep(5)
                    continue
                d = r.json()
                for x in d["results"]:
                    y = label(x["metadata"])
                    if y is not None and x["metadata"].get("acquisition", {}).get("image_type") == "dermoscopic":
                        rows.append(
                            {
                                "id": x["isic_id"],
                                "y": y,
                                "lesion": x["metadata"]["clinical"].get("lesion_id"),
                                "dx": x["metadata"]["clinical"].get("diagnosis_3"),
                                "url": x["files"]["thumbnail_256"]["url"],
                            }
                        )
                url = d.get("next")
        index.write_text(json.dumps(rows))
    seen, by = set(), {0: [], 1: []}
    random.seed(42)
    random.shuffle(rows)
    for r in rows:
        if r["lesion"] in seen:
            continue
        seen.add(r["lesion"])
        by[r["y"]].append(r)
    pick = by[0][:PER_CLASS] + by[1][:PER_CLASS]
    print({k: len(v) for k, v in by.items()}, "picked", len(pick), flush=True)
    with httpx.Client(timeout=60) as c:
        for r in pick:
            f = OUT / f"{r['id']}.jpg"
            if not f.exists():
                f.write_bytes(c.get(r["url"]).content)
    (OUT / "picked.json").write_text(json.dumps(pick))


if __name__ == "__main__":
    main()
