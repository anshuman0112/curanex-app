"""New survey screeners built with the fresh-test procedure from the start (2026-11-10). From backend/.

Rules fixed before 2024 was scored: logistic regression and boosting fitted on the BRFSS 2022 training rows; the higher AUC
on 2023 wins; threshold for 2023 sensitivity >= 0.82; 2024 scored once. Verified at sensitivity and AUC >= 0.80, live at
>= 0.75, otherwise off. Run: python training/train_brfss_fresh.py hearing_difficulty
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import training.train_brfss as tb  # noqa: E402

tb.RENAME["_RFDRHV9"] = "_RFDRHV8"


def main() -> None:
    d22, d23, d24 = tb.extract(2022), tb.extract(2023), tb.extract(2024)
    for t in sys.argv[1:]:
        feats = tb.features_for(t)
        a, v, b = (d[d[t].notna()] for d in (d22, d23, d24))
        x, y = a[feats].to_numpy(np.float32), a[t].to_numpy(int)
        tr, _ = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
        xv, yv = v[feats].to_numpy(np.float32), v[t].to_numpy(int)
        fits = {"logistic": tb.lr().fit(x[tr], y[tr]), "boosting": tb.gb().fit(x[tr], y[tr])}
        aucs = {k: roc_auc_score(yv, m.predict_proba(xv)[:, 1]) for k, m in fits.items()}
        name = max(aucs, key=aucs.get)
        model = fits[name]
        pv = model.predict_proba(xv)[:, 1]
        q = np.sort(pv[yv == 1])
        thr = float(q[int(np.floor(0.18 * len(q)))])
        v_report = tb.report(yv, pv, thr)
        test = tb.report(b[t].to_numpy(int), model.predict_proba(b[feats].to_numpy(np.float32))[:, 1], thr)
        status = (
            "verified"
            if min(test["sensitivity"], test["auc"]) >= 0.80
            else "live"
            if min(test["sensitivity"], test["auc"]) >= 0.75
            else "off"
        )
        print(f"{t}: {name} 2023 {v_report} | 2024 (once) {test} -> {status}", flush=True)
        meta = {
            "target": t,
            "title": tb.TARGETS[t],
            "features": feats,
            "model": name,
            "threshold": thr,
            "passed": status == "verified",
            "metrics": {"validation_auc": round(aucs[name], 3), "test_2023": v_report, "test_2024": test | {"kind": "first test"}},
            "status_2024": status,
            "trained_on": f"US CDC BRFSS 2022 survey, {len(tr):,} adults",
            "test_set": "BRFSS 2024, different people",
        }
        if name == "logistic":
            imp, sc, clf = (model.steps[i][1] for i in range(3))
            meta |= {
                "median": imp.statistics_.tolist(),
                "mean": sc.mean_.tolist(),
                "scale": sc.scale_.tolist(),
                "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]),
            }
        else:
            from skl2onnx import to_onnx

            onx = to_onnx(model, x[:1], options={id(model): {"zipmap": False}}, target_opset=17)
            (tb.OUT / f"{t}.onnx").write_bytes(onx.SerializeToString())
            meta |= {
                "onnx_sha256": hashlib.sha256((tb.OUT / f"{t}.onnx").read_bytes()).hexdigest(),
                "median": np.nanmedian(x[tr], axis=0).tolist(),
            }
        (tb.OUT / f"{t}.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
