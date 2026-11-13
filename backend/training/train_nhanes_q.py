"""NHANES round Q (2026-11-14): third replacement, after low testosterone failed in round P. Run from backend/.

Rules fixed before scoring (same procedure as training/train_nhanes_p.py):
- heart_damage_troponin: high-sensitivity cardiac troponin T (SSTNT) >= 14 ng/L (the Roche 99th-percentile upper reference
  limit), age 20+. Inputs: age, sex, BMI, told high BP, told diabetes, heart attack or coronary disease, heart failure, stroke,
  smoking. Train NHANES 1999-2000 + 2001-02; outside test on 2003-04 (cycle from SEQN), scored once.
- Logistic unless boosting wins by > 0.02 validation AUC; threshold = highest with validation sensitivity >= 0.85;
  Verified only if test sensitivity >= 0.80 and AUC >= 0.80.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_nhanes_p as P  # noqa: E402

RESULT = P.DIR / "results_q.json"


def main() -> None:
    if RESULT.exists():
        sys.exit("Round Q already scored once.")
    t = pd.read_sas(P.DIR / "SSTROP_A.xpt", format="xport", encoding="latin-1")[["SEQN", "SSTNT"]]
    parts = []
    for c in ("A", "B", "C"):
        d = P.merged(c, ["BMX", "SMQ", "DIQ", "BPQ", "MCQ"]).merge(t, on="SEQN", how="inner")
        f = P.common(d)
        f["y"] = np.where(d.SSTNT.notna() & (d.RIDAGEYR >= 20), (d.SSTNT >= 14).astype(float), np.nan)
        parts.append(f[["age", "male", "bmi", "high_bp", "diabetes", "heart_attack_or_chd", "heart_failure", "stroke", "smoker_current", "y"]])
    res = P.run("heart_damage_troponin", pd.concat(parts[:2]), parts[2], "1999-2002", "2003-04", 20)
    RESULT.write_text(json.dumps({"heart_damage_troponin": res}, indent=1))


if __name__ == "__main__":
    main()
