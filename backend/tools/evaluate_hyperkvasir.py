"""Outside test of the oesophagitis and endoscopy models on HyperKvasir (Baerum Hospital, Norway, 2020). From backend/.

Rules fixed before scoring: HyperKvasir images that match any Kvasir image (identical file, or a 64-bit difference hash within
6 bits) are removed first, because Kvasir came from the same hospital system and trained these models. Live models are scored
unchanged, once. Verified at sensitivity and AUC >= 0.80; still live at >= 0.75; otherwise off.
- oesophagitis_check: oesophagitis grades A and B-D against normal Z-line.
- endoscopy_check: colon polyps and ulcerative colitis (all grades), each against normal cecum.
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
HK = BACKEND / "data" / "eval" / "hyperkvasir" / "images"
KVASIR = BACKEND / "data" / "eval" / "kvasir" / "x" / "kvasir-dataset"


def dhash(path: Path) -> int:
    a = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.float32)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).ravel()), 2)


def kvasir_index() -> tuple[set, np.ndarray]:
    md5s, hashes = set(), []
    for f in KVASIR.rglob("*.jpg"):
        md5s.add(hashlib.md5(f.read_bytes()).hexdigest())  # noqa: S324 finds copies, not security
        hashes.append(dhash(f))
    return md5s, np.array(hashes, dtype=np.uint64)


def keep(files: list[Path], md5s: set, hashes: np.ndarray) -> list[Path]:
    out = []
    for f in files:
        if hashlib.md5(f.read_bytes()).hexdigest() in md5s:  # noqa: S324
            continue
        h = np.uint64(dhash(f))
        if (np.unpackbits(np.bitwise_xor(hashes, h).view(np.uint8).reshape(-1, 8), axis=1).sum(1) <= 6).any():
            continue
        out.append(f)
    return out


def score(adapter, files: list[Path], key: str) -> np.ndarray:
    s, flagged = [], []
    for f in files:
        p = adapter.predict({"image_b64": base64.b64encode(f.read_bytes()).decode()})
        s.append(p.probabilities[key])
        flagged.append(p.label.startswith("Possible") and key.lower() in p.label.lower())
    return np.array(s), np.array(flagged)


def score_raw(adapter, files: list[Path], names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Raw class scores from the deployed model; flagged when any named class reaches its deployed threshold."""
    labels, thr = adapter.meta["labels"], adapter.meta["thresholds"]
    idx = [labels.index(n) for n in names]
    s, flagged = [], []
    for f in files:
        p = adapter.session.run(None, {"image": adapter.prep(Image.open(f).convert("RGB"))})[0][0]
        s.append(max(p[i] for i in idx))
        flagged.append(any(p[i] >= thr[i] for i in idx))
    return np.array(s), np.array(flagged)


def result(pos, neg) -> dict:
    (sp, fp), (sn, fn) = pos, neg
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    return {
        "n": int(len(y)),
        "positives": int(len(sp)),
        "sensitivity": round(float(fp.mean()), 3),
        "specificity": round(float(1 - fn.mean()), 3),
        "auc": round(float(roc_auc_score(y, np.r_[sp, sn])), 3),
    }


def status(r: dict) -> str:
    return "verified" if min(r["sensitivity"], r["auc"]) >= 0.80 else "live" if min(r["sensitivity"], r["auc"]) >= 0.75 else "off"


def main(which: str) -> None:
    md5s, hashes = kvasir_index()
    files = lambda *names: [f for n in names for f in sorted((HK / n).rglob("*.jpg"))]  # noqa: E731
    out = {}
    if which == "esophagitis":
        from workers.torch.adapters.esophagitis_check import Adapter

        a = Adapter()
        a.load()
        pos, neg = keep(files("esophagitis-a", "esophagitis-b-d"), md5s, hashes), keep(files("z-line"), md5s, hashes)
        print("after removing Kvasir copies:", len(pos), "oesophagitis,", len(neg), "normal Z-line", flush=True)
        key = a.titles["Esophagitis"]
        out["esophagitis_check"] = result(score(a, pos, key), score(a, neg, key))
    else:
        from workers.torch.adapters.endoscopy_check import Adapter

        a = Adapter()
        a.load()
        neg = keep(files("cecum"), md5s, hashes)
        for name, folders, classes in (
            ("colon polyps", ("polyps",), ("polyps", "dyed-lifted-polyps")),
            ("ulcerative colitis", tuple(p.name for p in HK.glob("uc-*")), ("ulcerative-colitis",)),
        ):
            pos = keep(files(*folders), md5s, hashes)
            print(name, "after removing Kvasir copies:", len(pos), "against", len(neg), "normal cecum", flush=True)
            out[f"endoscopy_check: {name}"] = result(score_raw(a, pos, classes), score_raw(a, neg, classes))
    for k, r in out.items():
        r["status"] = status(r)
        print(k, r, flush=True)
    path = BACKEND / "data" / "eval" / "hyperkvasir" / f"result_{which}.json"
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
