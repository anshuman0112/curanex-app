"""Convert RSNA Pneumonia DICOM training images to PNG (pydicom; data only). Run from backend/ in the local venv:

  python tools/rsna_to_png.py

Reads data/eval/rsna/stage_2_train_images/*.dcm, writes data/eval/rsna/png/<patientId>.png and labels.csv
(patientId, target) from stage_2_detailed_class_info.csv: "Lung Opacity" = 1, anything else = 0.
"""
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "eval" / "rsna"


def convert(path: Path) -> bool:
    out = ROOT / "png" / f"{path.stem}.png"
    if out.exists():
        return True
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0) * 255
        Image.fromarray(arr.astype(np.uint8)).save(out)
        return True
    except Exception:  # unreadable DICOM: skipped, counted below
        return False


def main() -> None:
    (ROOT / "png").mkdir(exist_ok=True)
    labels = {}
    with (ROOT / "stage_2_detailed_class_info.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels[r["patientId"]] = int(r["class"] == "Lung Opacity")
    paths = sorted((ROOT / "stage_2_train_images").glob("*.dcm"))
    with ThreadPoolExecutor(8) as pool:
        done = sum(pool.map(convert, paths))
    with (ROOT / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patientId", "target"])
        w.writerows([p, t] for p, t in labels.items() if (ROOT / "png" / f"{p}.png").exists())
    print(f"converted {done} of {len(paths)}; positives {sum(labels.values())} of {len(labels)}")


if __name__ == "__main__":
    main()
