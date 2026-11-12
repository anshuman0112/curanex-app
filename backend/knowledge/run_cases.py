"""Five-case end-to-end benchmark of the doctor module (live AI + library). Cases fixed before running (2026-11-12).
From backend/: python -m knowledge.run_cases   ->  backend/data/knowledge/five_cases.json  (every answer + timings)
Run once with an empty cache (fresh timings), once again to measure the cache.
"""

import json
import time

from api.knowledge import answer, cache
from knowledge.common import DATA

FULL = {"weight_kg": 65, "kidney_disease": False, "liver_disease": False, "allergies": [], "current_medicines": []}
CASES = [
    {"name": "Fever, cough, breathlessness", "patient": {"age": 45, "sex": "male", **FULL},
     "symptoms": "fever for 3 days, a lot of cough with yellow sputum, difficulty breathing",
     "report": {"WBC": "15,800 per microlitre (high)", "SpO2": "93%", "Chest X-ray": "right lower lobe consolidation"}},
    {"name": "Thirst, frequent urination", "patient": {"age": 52, "sex": "male", **FULL, "weight_kg": 88},
     "symptoms": "always thirsty, passing urine many times at night, tired for 2 months",
     "report": {"Fasting glucose": "168 mg/dL", "HbA1c": "8.1%", "Creatinine": "0.9 mg/dL"}},
    {"name": "Long cough, weight loss", "patient": {"age": 34, "sex": "male", **FULL, "weight_kg": 52},
     "symptoms": "cough for 5 weeks, evening fever, night sweats, weight loss",
     "report": {"Sputum CBNAAT": "Mycobacterium tuberculosis detected, rifampicin resistance not detected", "Chest X-ray": "right upper lobe cavity"}},
    {"name": "Tired, pale, breathless", "patient": {"age": 28, "sex": "female", "pregnant": False, **FULL, "weight_kg": 48},
     "symptoms": "tired all the time, pale, breathless on climbing stairs, heavy periods",
     "report": {"Haemoglobin": "8.2 g/dL", "MCV": "68 fL", "Ferritin": "6 ng/mL"}},
    {"name": "Swelling, high creatinine", "patient": {"age": 61, "sex": "male", **FULL, "kidney_disease": True, "weight_kg": 70},
     "symptoms": "swelling of both feet, passing less urine, long history of diabetes and high blood pressure",
     "report": {"Creatinine": "2.6 mg/dL", "eGFR": "27 mL/min/1.73m2", "Urine albumin/creatinine": "480 mg/g", "Potassium": "5.6 mmol/L"}},
]


def timed(f, *a):
    t = time.time()
    out = f(*a)
    return out, round(time.time() - t, 1)


def main() -> None:
    results = []
    for c in CASES:
        case = {k: v for k, v in c.items() if k != "name"}
        tests, t1 = timed(answer.suggest_tests, case)
        diff, t2 = timed(answer.differential, case)
        top = diff["supported"][0]["disease"] if diff["supported"] else None
        treat, t3 = timed(answer.treatment, case, top) if top else ({}, 0.0)
        results.append({"case": c["name"], "seconds": {"tests": t1, "differential": t2, "treatment": t3},
                        "tests": tests, "differential": diff, "treatment": treat})
        print(f"{c['name']}: tests {len(tests['tests'])} ({t1}s) | supported {[d['disease'] for d in diff['supported']][:4]} also {diff['also_consider']} ({t2}s) | "
              f"plan {len(treat.get('plan', []))} meds {[m['label']['medicine'] for m in treat.get('medicines', [])]} ({t3}s) | "
              f"dropped {tests['dropped']}/{diff['dropped']}/{treat.get('dropped', 0)} | emergency {tests['emergency']}", flush=True)
    (DATA / "five_cases.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("cache", cache.stats)


if __name__ == "__main__":
    main()
