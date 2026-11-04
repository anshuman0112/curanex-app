"""Cache model features for the RSNA PNGs (see rsna_to_png.py). Worker image, no network; same mounts as
extract_vinbig_features.py."""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tools")
from extract_xrv_features import init, one  # noqa: E402

ROOT = Path("/data/rsna")


def main() -> None:
    with (ROOT / "labels.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with Pool(4, initializer=init) as pool:
        out = pool.map(one, [str(ROOT / "png" / f"{r['patientId']}.png") for r in rows], chunksize=16)
    keep = [i for i, r in enumerate(out) if r is not None]
    np.savez(Path("/data/features_rsna.npz"), X=np.stack([out[i][0] for i in keep]), scores=np.stack([out[i][1] for i in keep]),
             y=np.array([int(rows[i]["target"]) for i in keep]))
    print(f"saved {len(keep)} of {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
