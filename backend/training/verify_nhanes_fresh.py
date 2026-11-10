"""Fresh test of the NHANES screeners on survey cycles that were never used (2015-16 and 2009-10). From backend/.

Rules fixed before any fresh cycle was scored (2026-11-10), the same as training/verify_brfss_2024.py:
- Verified models: scored unchanged on the fresh cycle as a confirmation (lose Verified below 0.80, go off below 0.75).
- Unverified models: their old test cycle becomes validation. Logistic and boosting are both fitted on the original training
  rows; the higher validation AUC wins, threshold at validation sensitivity >= 0.82; the fresh cycle is scored once.
  Verified at >= 0.80 sensitivity and AUC, live at >= 0.75, otherwise off.
- Fatty liver has no other FibroScan cycle, so it can't be re-tested.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import training.train_nhanes as tn  # noqa: E402

tn.CYCLE_YEARS.update({"F": "2009-10", "I": "2015-16"})
FRESH = {
    "iron_deficiency": "I",
    "heart_failure": "I",
    "kidney_disease_lab": "I",
    "gout": "I",
    "osteoporosis": "F",
    "low_bone_density": "F",
    "hearing_loss": "F",
}


def score_saved(meta: dict, x: np.ndarray) -> np.ndarray:
    x = np.where(np.isnan(x), np.array(meta["impute"]), x)
    z = ((x - np.array(meta["mean"])) / np.array(meta["scale"])) @ np.array(meta["coef"]) + meta["intercept"]
    return 1 / (1 + np.exp(-z))


def rows(df, target, feats, min_age, women_12_49=False):
    d = df[df[target].notna() & (df["age"] >= min_age)]
    if women_12_49:
        d = d[(d["male"] == 0) & (d["age"] <= 49)]
    return d[feats].to_numpy(float), d[target].to_numpy(int)


def main() -> None:
    cache = {}
    out = {}
    for target, fresh in FRESH.items():
        path = tn.OUT / f"{target}.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        src, dst, feats, min_age = tn.TARGETS[target]
        for c in (src, dst, fresh):
            if c not in cache:
                cache[c] = tn.cycle(c)
        old = meta["metrics"]["test"]
        women = target == "iron_deficiency"  # offered only to women and girls 12-49, so tested on them
        xf, yf = rows(cache[fresh], target, feats, min_age, women)
        if target == "hearing_loss":
            keep = cache[fresh].loc[cache[fresh][target].notna() & (cache[fresh]["age"] >= min_age), "age"].to_numpy() <= 69
            xf, yf = xf[keep], yf[keep]
        if len(yf) < 50 or yf.sum() < 10:
            print(target, "fresh cycle has too few cases:", len(yf), int(yf.sum()), flush=True)
            continue
        if old["sensitivity"] >= 0.80 and old["auc"] >= 0.80:
            test = tn.report(yf, score_saved(meta, xf), meta["threshold"])
            kind = "confirmation"
        else:
            x, y = rows(cache[src], target, feats, min_age)
            tr, _ = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
            xv, yv = rows(cache[dst], target, feats, min_age)
            fits = {"logistic": tn.lr().fit(x[tr], y[tr]), "boosting": tn.gb().fit(x[tr], y[tr])}
            aucs = {k: roc_auc_score(yv, m.predict_proba(xv)[:, 1]) for k, m in fits.items()}
            name = max(aucs, key=aucs.get)
            pv = fits[name].predict_proba(xv)[:, 1]
            q = np.sort(pv[yv == 1])
            thr = float(q[int(np.floor(0.18 * len(q)))])
            test = tn.report(yf, fits[name].predict_proba(xf)[:, 1], thr)
            kind = f"retuned on {tn.CYCLE_YEARS[dst]} ({name})"
            if name == "logistic" and test["sensitivity"] >= 0.75 and test["auc"] >= 0.75:
                imp, sc, clf = (fits[name].steps[i][1] for i in range(3))
                meta |= {
                    "threshold": thr,
                    "impute": imp.statistics_.tolist(),
                    "mean": sc.mean_.tolist(),
                    "scale": sc.scale_.tolist(),
                    "coef": clf.coef_[0].tolist(),
                    "intercept": float(clf.intercept_[0]),
                }
            elif name == "boosting":
                kind += " - boosting won but the adapter is logistic-only, so the live model is unchanged"
        status = (
            "verified"
            if test["sensitivity"] >= 0.80 and test["auc"] >= 0.80
            else "live"
            if min(test["sensitivity"], test["auc"]) >= 0.75
            else "off"
        )
        if "boosting won" in kind:
            status = "unchanged (boosting not deployable)"
        meta["metrics"]["test_fresh"] = test | {"cycle": tn.CYCLE_YEARS[fresh], "kind": kind}
        meta["status_fresh"] = status
        path.write_text(json.dumps(meta, indent=1))
        out[target] = meta["metrics"]["test_fresh"] | {"status": status}
        print(f"{target:18s} {kind:42s} {tn.CYCLE_YEARS[fresh]} {test} -> {status}", flush=True)
    (BACKEND / "data" / "eval" / "nhanes" / "verify_fresh.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
