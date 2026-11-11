"""Third test round for the Unverified survey screeners, on BRFSS 2025 (never used). From backend/.

Rules fixed before 2025 was scored (2026-11-11):
- Train on more years: 2022 + 2023 (blood pressure: 2021 + 2023, the odd years that ask it).
- Candidates: logistic regression and a larger gradient boosting model (depth 6, 400 trees). Chosen by AUC on 2024
  (blood pressure: a 15% held-out part of its training years, since 2024 does not ask it). Threshold for sensitivity >= 0.82
  on that validation data.
- 2025 scored once. Verified at sensitivity and AUC >= 0.80; live at >= 0.75; otherwise off. A new model replaces the live one
  only if it is at least live; otherwise the live model and its earlier result stay.
Run: python training/verify_brfss_2025.py cancer skin_cancer vision_loss hearing_difficulty high_blood_pressure
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import training.train_brfss as tb  # noqa: E402

tb.RENAME["_RFDRHV9"] = "_RFDRHV8"


def big_gb():
    return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42)


def main() -> None:
    years = {y: tb.extract(y) for y in (2021, 2022, 2023, 2024, 2025)}
    print({y: len(d) for y, d in years.items()}, flush=True)
    for t in sys.argv[1:]:
        path = tb.OUT / f"{t}.json"
        old = json.loads(path.read_text(encoding="utf-8"))
        feats = old["features"]
        odd = t in tb.ODD_YEAR
        train_years = (2021, 2023) if odd else (2022, 2023)
        a = pd.concat([years[y][years[y][t].notna()] for y in train_years], ignore_index=True)
        x, y = a[feats].to_numpy(np.float32), a[t].to_numpy(int)
        if odd:
            tr, va = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
            xv, yv, xt, yt = x[va], y[va], x[tr], y[tr]
        else:
            v = years[2024][years[2024][t].notna()]
            xv, yv, xt, yt = v[feats].to_numpy(np.float32), v[t].to_numpy(int), x, y
        fits = {"logistic": tb.lr().fit(xt, yt), "boosting": big_gb().fit(xt, yt)}
        aucs = {k: roc_auc_score(yv, m.predict_proba(xv)[:, 1]) for k, m in fits.items()}
        name = max(aucs, key=aucs.get)
        model = fits[name]
        pv = model.predict_proba(xv)[:, 1]
        q = np.sort(pv[yv == 1])
        thr = float(q[int(np.floor(0.18 * len(q)))])
        b = years[2025][years[2025][t].notna()]
        test = tb.report(b[t].to_numpy(int), model.predict_proba(b[feats].to_numpy(np.float32))[:, 1], thr)
        low = min(test["sensitivity"], test["auc"])
        status = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
        print(f"{t:20s} {name:8s} validation AUC {aucs[name]:.3f} | 2025 (once) {test} -> {status}", flush=True)
        old.setdefault("history", []).append({"round": "2025", "model": name, "test_2025": test, "status": status})
        if status != "off":
            old.update(
                {
                    "model": name,
                    "threshold": thr,
                    "trained_on": f"US CDC BRFSS {' + '.join(map(str, train_years))} survey, {len(yt):,} adults",
                }
            )
            old["metrics"]["test_2025"] = test | {
                "kind": "retrained on more years, chosen on 2024" if not odd else "retrained on 2021 + 2023"
            }
            if name == "logistic":
                imp, sc, clf = (model.steps[i][1] for i in range(3))
                old.update(
                    {
                        "median": imp.statistics_.tolist(),
                        "mean": sc.mean_.tolist(),
                        "scale": sc.scale_.tolist(),
                        "coef": clf.coef_[0].tolist(),
                        "intercept": float(clf.intercept_[0]),
                    }
                )
                old.pop("onnx_sha256", None)
            else:
                from skl2onnx import to_onnx

                onx = to_onnx(model, xt[:1], options={id(model): {"zipmap": False}}, target_opset=17)
                (tb.OUT / f"{t}.onnx").write_bytes(onx.SerializeToString())
                old["onnx_sha256"] = hashlib.sha256((tb.OUT / f"{t}.onnx").read_bytes()).hexdigest()
                old["median"] = np.nanmedian(xt, axis=0).tolist()
                for k in ("mean", "scale", "coef", "intercept"):
                    old.pop(k, None)
        path.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
