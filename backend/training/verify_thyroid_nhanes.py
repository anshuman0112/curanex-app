"""Outside test for the underactive thyroid check on US CDC NHANES 2007-2012 (real people, USA, never used in training,
which was the UCI Garavan Institute data from Australia in the 1980s). Run from backend/.

Rules fixed before the NHANES thyroid data was downloaded (written 2026-11-13):
- People: age 18+, not pregnant, TSH and free T4 both measured, taking no thyroid medicine
  (levothyroxine, liothyronine, thyroid desiccated, methimazole, propylthiouracil) in the past 30 days.
- Truth: free T4 below the lower reference limit, LBXT4F < 0.6 ng/dL. This is a different blood test from the
  model's input (TSH), so the test is not circular: it asks whether the TSH check finds people whose thyroid hormone
  is actually low.
- Inputs, exactly as the app sends them: TSH (LBXTSH1, mIU/L), age, sex, on_thyroxine = "No". T3/T4 left at the
  training medians, as in the app.
- The live model is scored unchanged, once, at its stored threshold. Pass = sensitivity >= 0.80 and AUC >= 0.80,
  with at least 20 people with low free T4; fewer than 20 = inconclusive.
- Result written to data/eval/nhanes_thyroid/result.json; the script refuses to score twice.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

BACKEND = Path(__file__).resolve().parents[1]
DIR = BACKEND / "data" / "eval" / "nhanes_thyroid"
RESULT = DIR / "result.json"
META = BACKEND / "workers" / "tabular" / "models" / "cilab_ufersa_thyroid_disease_ai" / "model_meta.json"
CYCLES = {"E": 2007, "F": 2009, "G": 2011}
THYROID_DRUGS = ("LEVOTHYROXINE", "LIOTHYRONINE", "THYROID", "METHIMAZOLE", "PROPYLTHIOURACIL", "LIOTRIX")


def get(name: str, c: str) -> pd.DataFrame:
    path = DIR / f"{name}_{c}.xpt"
    if not path.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{CYCLES[c]}/DataFiles/{name}_{c}.xpt"
        urllib.request.urlretrieve(url, path)
    return pd.read_sas(path, format="xport", encoding="latin-1")


def people() -> pd.DataFrame:
    rows = []
    for c in CYCLES:
        d = get("DEMO", c)[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDEXPRG"]]
        t = get("THYROD", c)[["SEQN", "LBXTSH1", "LBXT4F"]]
        rx = get("RXQ_RX", c)
        drug = rx["RXDDRUG"].fillna("").str.upper()
        treated = set(rx.loc[drug.apply(lambda s: any(k in s for k in THYROID_DRUGS)), "SEQN"])
        m = d.merge(t, on="SEQN")
        m = m[(m.RIDAGEYR >= 18) & (m.RIDEXPRG != 1) & m.LBXTSH1.notna() & m.LBXT4F.notna() & ~m.SEQN.isin(treated)]
        rows.append(m.assign(cycle=c))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    if RESULT.exists():
        sys.exit(f"Already scored once: {RESULT}")
    p_ = people()
    y = (p_.LBXT4F < 0.6).astype(int).to_numpy()
    m = json.loads(META.read_text())
    u = m["models"]["underactive"]
    rows = []
    for _, r in p_.iterrows():
        known = {"log_tsh": float(np.log1p(r.LBXTSH1)), "on_thyroxine": 0.0, "on_antithyroid": 0.0,
                 "age": float(r.RIDAGEYR), "female": float(r.RIAGENDR == 2)}
        rows.append([known.get(f, md) for f, md in zip(m["features"], u["median"], strict=True)])
    x = np.array(rows)
    d = ((x - np.array(u["mean"])) / np.array(u["scale"])) @ np.array(u["coef"]) + u["intercept"]
    p = 1 / (1 + np.exp(-d))
    tn, fp, fn, tp = confusion_matrix(y, p >= u["threshold"], labels=[0, 1]).ravel()
    res = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
           "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, p)), 3)}
    res["passed"] = bool(res["positives"] >= 20 and res["sensitivity"] >= 0.80 and res["auc"] >= 0.80)
    res["inconclusive"] = res["positives"] < 20
    res |= {"set": "NHANES 2007-08, 2009-10, 2011-12 (USA), adults, not pregnant, no thyroid medicine",
            "truth": "free T4 < 0.6 ng/dL", "model": "live model, unchanged, TSH + age + sex as in the app", "scored": "2026-11-13"}
    RESULT.write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
