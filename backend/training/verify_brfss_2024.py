"""Second, fresh test of the survey screeners on BRFSS 2024 (released after the models were built). From backend/.

Rules fixed before any 2024 data was scored (2026-11-10):
- Models already Verified on 2023: scored unchanged on 2024 as a confirmation. They lose Verified if 2024 sensitivity or
  AUC falls below 0.80, and go off if either falls below 0.75.
- Models live but Unverified on 2023: 2023 becomes validation. Logistic regression and boosting are both fitted on the same
  2022 training rows; the one with the higher 2023 AUC is kept, with the threshold giving 2023 sensitivity >= 0.82. Then 2024
  is scored once: Verified if sensitivity and AUC >= 0.80; still live if both >= 0.75; otherwise switched off.
- Blood pressure and cholesterol are not asked in even years, so they cannot be tested on 2024.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import training.train_brfss as tb  # noqa: E402

tb.RENAME["_RFDRHV9"] = "_RFDRHV8"  # same question, renamed in 2024
OUT = tb.OUT


def score_saved(meta: dict, x: np.ndarray) -> np.ndarray:
    if meta["model"] == "logistic":
        x = np.where(np.isnan(x), np.array(meta["median"]), x)
        z = ((x - np.array(meta["mean"])) / np.array(meta["scale"])) @ np.array(meta["coef"]) + meta["intercept"]
        return 1 / (1 + np.exp(-z))
    s = ort.InferenceSession(str(OUT / f"{meta['target']}.onnx"), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x.astype(np.float32)})[1][:, 1]


def threshold_for(y: np.ndarray, p: np.ndarray, sens: float) -> float:
    q = np.sort(p[y == 1])
    return float(q[int(np.floor((1 - sens) * len(q)))])


def main() -> None:
    d22, d23, d24 = tb.extract(2022), tb.extract(2023), tb.extract(2024)
    print(f"2024: {len(d24):,} people", flush=True)
    results = {}
    for path in sorted(OUT.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        t = meta["target"]
        if t in tb.ODD_YEAR or "test_2023" not in meta["metrics"]:
            continue
        old = meta["metrics"]["test_2023"]
        live = old["sensitivity"] >= 0.75 and old["auc"] >= 0.75
        verified = old["sensitivity"] >= 0.80 and old["auc"] >= 0.80
        if not live:
            continue
        feats = meta["features"]
        b = d24[d24[t].notna()]
        x24, y24 = b[feats].to_numpy(np.float32), b[t].to_numpy(int)
        if verified:
            test = tb.report(y24, score_saved(meta, x24), meta["threshold"])
            kind = "confirmation"
        else:
            a = d22[d22[t].notna()]
            x, y = a[feats].to_numpy(np.float32), a[t].to_numpy(int)
            tr, _ = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)  # same training rows as before
            v = d23[d23[t].notna()]
            xv, yv = v[feats].to_numpy(np.float32), v[t].to_numpy(int)
            fits = {"logistic": tb.lr().fit(x[tr], y[tr]), "boosting": tb.gb().fit(x[tr], y[tr])}
            aucs = {k: roc_auc_score(yv, m.predict_proba(xv)[:, 1]) for k, m in fits.items()}
            name = max(aucs, key=aucs.get)
            model = fits[name]
            thr = threshold_for(yv, model.predict_proba(xv)[:, 1], 0.82)
            test = tb.report(y24, model.predict_proba(x24)[:, 1], thr)
            kind = "retuned on 2023"
            new = {"model": name, "threshold": thr, "validation_2023_auc": round(aucs[name], 3)}
            if name == "logistic":
                imp, sc, clf = (model.steps[i][1] for i in range(3))
                new |= {
                    "median": imp.statistics_.tolist(),
                    "mean": sc.mean_.tolist(),
                    "scale": sc.scale_.tolist(),
                    "coef": clf.coef_[0].tolist(),
                    "intercept": float(clf.intercept_[0]),
                }
            else:
                from skl2onnx import to_onnx

                onx = to_onnx(model, x[:1], options={id(model): {"zipmap": False}}, target_opset=17)
                (OUT / f"{t}.onnx").write_bytes(onx.SerializeToString())
                new |= {
                    "onnx_sha256": hashlib.sha256((OUT / f"{t}.onnx").read_bytes()).hexdigest(),
                    "median": np.nanmedian(x[tr], axis=0).tolist(),
                }
                for k in ("mean", "scale", "coef", "intercept"):
                    meta.pop(k, None)
            if test["sensitivity"] >= 0.75 and test["auc"] >= 0.75:
                meta.update(new)  # the retuned model replaces the old one only if it stays live
        status = (
            "verified"
            if test["sensitivity"] >= 0.80 and test["auc"] >= 0.80
            else "live"
            if test["sensitivity"] >= 0.75 and test["auc"] >= 0.75
            else "off"
        )
        meta["metrics"]["test_2024"] = test | {"kind": kind}
        meta["status_2024"] = status
        path.write_text(json.dumps(meta, indent=1))
        results[t] = (kind, test, status)
        print(f"{t:18s} {kind:16s} 2024 {test} -> {status}", flush=True)
    (BACKEND / "data" / "eval" / "brfss" / "verify_2024.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
