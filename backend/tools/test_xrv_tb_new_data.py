"""Second pass for the TB layer: python tools/test_xrv_tb_new_data.py (from backend/).

The first threshold, set on a tiny validation split, was too strict (test sensitivity 0.763). That test can't be reused, so:
threshold comes from 5-fold out-of-fold scores on the NLM training and validation rows only (highest with sensitivity >= 0.85),
the model is refit on those rows with the C already chosen, and new data is scored once: TBX11K (China, CC BY 4.0),
all TB images against equal numbers of healthy people and of sick people without TB.
"""
import csv
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from tools.extract_fast import run  # noqa: E402
from tools.train_xrv_tb_head import CACHE, OUT, TARGET, report  # noqa: E402

ROOT = BACKEND / "data" / "eval" / "tbx"
FEATS = BACKEND / "data" / "eval" / "features_tbx.npz"


def tbx():
    if FEATS.exists():
        d = np.load(FEATS, allow_pickle=False)
        return d["X"], d["kind"]
    zip_path = ROOT / "data.zip"
    with zipfile.ZipFile(zip_path) as z:
        rows = list(csv.DictReader(z.read("tbx11k-simplified/data.csv").decode().splitlines()))
        seen, by = set(), {"tb": [], "healthy": [], "sick_but_no_tb": []}
        for r in rows:
            if r["fname"] not in seen and r["image_type"] in by:
                seen.add(r["fname"])
                by[r["image_type"]].append(r["fname"])
        print({k: len(v) for k, v in by.items()}, flush=True)
        n = len(by["tb"])
        rng = random.Random(42)  # noqa: S311 sampling, not security
        pick = [(f, "tb") for f in by["tb"]] + [(f, k) for k in ("healthy", "sick_but_no_tb") for f in rng.sample(sorted(by[k]), n)]
        for f, _ in pick:
            if not (ROOT / "tbx11k-simplified" / "images" / f).exists():
                z.extract(f"tbx11k-simplified/images/{f}", ROOT)
    out = run([str(ROOT / "tbx11k-simplified" / "images" / f) for f, _ in pick])
    keep = [i for i, r in enumerate(out) if r is not None]
    X, kind = np.stack([out[i][0] for i in keep]), np.array([pick[i][1] for i in keep])
    np.savez(FEATS, X=X, kind=kind)
    return X, kind


def main() -> None:
    d = np.load(CACHE, allow_pickle=False)
    X, y, source = d["X"], d["y"], d["source"]
    nlm = np.where(source == "nlm")[0]
    tr, rest = train_test_split(nlm, test_size=0.30, random_state=42, stratify=y[nlm])
    va, _ = train_test_split(rest, test_size=0.50, random_state=42, stratify=y[rest])
    use = np.r_[tr, va]
    c = json.loads(OUT.read_text())["metrics"]["C"]
    oof = np.zeros(len(use))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(use, y[use]):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000)).fit(X[use[a]], y[use[a]])
        oof[b] = m.predict_proba(X[use[b]])[:, 1]
    t = float(max(v for v in np.unique(oof) if (oof[y[use] == 1] >= v).mean() >= TARGET))
    print("out-of-fold threshold", t, flush=True)
    scaler = StandardScaler().fit(X[use])
    model = LogisticRegression(C=c, max_iter=3000).fit(scaler.transform(X[use]), y[use])
    Xt, kind = tbx()
    s = model.predict_proba(scaler.transform(Xt))[:, 1]
    yt = (kind == "tb").astype(int)
    non = {k: report(yt[(kind == "tb") | (kind == k)], s[(kind == "tb") | (kind == k)], t) for k in ("healthy", "sick_but_no_tb")}
    results = {"threshold": t, "C": c, "tbx11k_all": report(yt, s, t), "against": non, "first_pass": json.loads(OUT.read_text())["metrics"]}
    print(json.dumps({k: results[k] for k in ("threshold", "tbx11k_all", "against")}, indent=0), flush=True)
    OUT.write_text(json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist(),
                               "intercept": float(model.intercept_[0]), "threshold": t, "metrics": results}))


if __name__ == "__main__":
    main()
