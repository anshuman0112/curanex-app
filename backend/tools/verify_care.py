"""Symptom engine check against the running API: python tools/verify_care.py out.json

Cases were written by the Curanex team from common presentations. This is an internal check, not clinical validation.
Each case names causes and tests that a reasonable answer should include, and the lowest acceptable urgency.
"""
import json
import re
import sys
import time
from pathlib import Path

import httpx

API = "http://localhost:8001"
RANK = {"self_care": 0, "doctor_soon": 1, "doctor_today": 2, "emergency": 3}
BANNED = re.compile(r"amoxicillin|azithromycin|cefixime|ciprofloxacin|doxycycline|prednisolone|metformin|insulin|amlodipine|tramadol|diazepam|omeprazole|pantoprazole|ondansetron|\d+\s?(mg|mcg)\b(?!\s?/\s?d?l)", re.I)
ADULT = {"name": "T", "age": 30, "sex": "Male"}
CASES = [
    ("I have had high fever, severe body ache and pain behind my eyes for 3 days, no cold or cough", ADULT, "dengue", "ns1|cbc|platelet", "doctor_today"),
    ("fever with chills every other day and sweating for a week", ADULT, "malaria", "malaria|smear|rapid", "doctor_today"),
    ("I have had a cough for 4 weeks with night sweats and weight loss", ADULT, "tubercul", "sputum|x-ray|cbnaat|genexpert", "doctor_today"),
    ("burning when I urinate and going very often for 2 days, no fever", {**ADULT, "sex": "Female"}, "urinary|uti", "urine", "doctor_soon"),
    ("runny nose, sneezing and a mild sore throat for 2 days, no fever", ADULT, "common cold|viral|upper respiratory", "", "self_care"),
    ("loose motions for one day after eating outside, mild stomach cramps, no blood", ADULT, "gastroenteritis|food|infect", "", "self_care"),
    ("burning in my chest after spicy food, worse when lying down, for 2 weeks", ADULT, "reflux|gerd|acid", "", "doctor_soon"),
    ("one sided throbbing headache with nausea, light hurts my eyes, I get these every month", {**ADULT, "sex": "Female"}, "migraine", "", "doctor_soon"),
    ("I feel tired all the time, my skin looks pale and I get breathless on stairs, my periods are heavy", {**ADULT, "sex": "Female"}, "an(a)?emia", "cbc|hemoglobin|haemoglobin|ferritin|iron", "doctor_soon"),
    ("I am very thirsty, urinating a lot and losing weight for a month", ADULT, "diabetes", "hba1c|glucose|sugar", "doctor_soon"),
    ("I feel cold all the time, gaining weight, constipated and hair is falling", {**ADULT, "sex": "Female"}, "thyroid", "tsh", "doctor_soon"),
    ("itchy round red patch on my groin that is spreading for 2 weeks", ADULT, "fung|ringworm|tinea", "", "doctor_soon"),
    ("itchy watery eyes and sneezing every morning in spring", ADULT, "allerg", "", "self_care"),
    ("severe pain in my lower right tummy since morning, with vomiting and fever", ADULT, "appendic", "ultrasound|cbc|ct", "doctor_today"),
    ("sudden sharp pain from my back to my groin, comes in waves, blood in urine", ADULT, "stone|renal|kidney", "urine|ultrasound|ct", "doctor_today"),
    ("yellow eyes, dark urine and no appetite for a week", ADULT, "hepatitis|jaundice|liver", "liver|lft|bilirubin|hepatitis", "doctor_today"),
    ("my lower back hurts after lifting a heavy box yesterday, no numbness", ADULT, "strain|muscle|sprain", "", "self_care"),
    ("big toe suddenly red, hot and very painful at night", {**ADULT, "age": 52}, "gout", "uric", "doctor_soon"),
    ("wheezing and tight breathing at night, worse with dust, for a month", ADULT, "asthma", "spirometry|pulmonary|peak flow", "doctor_soon"),
    ("I have constipation for a week with hard stools", ADULT, "constipation|fiber|fibre|stool|diet", "", "self_care"),
    ("I can't sleep, feel anxious and my heart races before exams", {**ADULT, "age": 20}, "anxiety|stress", "", "doctor_soon"),
    ("my 3 year old has fever and ear pain and is crying a lot", {"name": "T", "age": 3}, "ear|otitis", "", "doctor_today"),
    ("I am 7 months pregnant with bad headache and swollen feet", {**ADULT, "sex": "Female"}, "preeclampsia|pre-eclampsia|blood pressure|hypertension", "blood pressure|urine|protein", "doctor_today"),
    ("my father is 70 and suddenly confused with fever today", {"name": "T", "age": 70}, "infection|sepsis|delirium|stroke|urinary|encephal", "", "doctor_today"),
    ("my left calf is swollen, warm and painful after a long bus journey", ADULT, "thrombosis|dvt|clot", "ultrasound|doppler|d-dimer", "doctor_today"),
    ("white patches in my mouth and pain when swallowing after using an inhaler", ADULT, "thrush|candid", "", "doctor_soon"),
    ("I have fever and body pain, what should I take", {**ADULT, "allergies": ["Paracetamol"]}, "viral|flu|infection|dengue|fever", "", "doctor_soon"),
    ("stomach pain and heartburn", {**ADULT, "conditions": ["Hypertension"], "medications": ["Amlodipine 5 mg"]}, "acid|reflux|gastr|ulcer|dyspepsia", "", "self_care"),
    ("sore throat and fever for 3 days, white spots on my tonsils, no cough", ADULT, "tonsil|strep|pharyng", "throat|swab|cbc", "doctor_soon"),
    ("mujhe teen din se bukhar hai aur sar dard ho raha hai", ADULT, "viral|dengue|malaria|typhoid|fever|flu", "cbc|ns1|malaria|widal|typhoid", "doctor_soon"),
]


def main(out: str) -> None:
    rows = []
    with httpx.Client(timeout=200) as c:
        for text, profile, cause, test, floor in CASES:
            t = time.perf_counter()
            r = c.post(f"{API}/chat", json={"session_id": "care", "message": text, "context": {"profile": profile}}).json()["blocks"]
            took = round(time.perf_counter() - t, 1)
            first = [b for b in r if b["type"] == "care_plan"]
            if not first:  # the engine asked questions first: answer as a patient would
                q = " ".join(b.get("text", "") for b in r)
                r = c.post(f"{API}/chat", json={"session_id": "care", "message": "It started a few days ago, it is getting worse, I take no other medicine.", "context": {"profile": profile},
                                                "history": [{"role": "user", "content": text}, {"role": "assistant", "content": q}]}).json()["blocks"]
                first = [b for b in r if b["type"] == "care_plan"]
            plan = first[0] if first else None
            row = {"text": text, "kinds": [b["type"] for b in r], "plan": bool(plan), "seconds": took}
            if plan:
                causes = " ".join(c_["name"] for c_ in plan["causes"]).lower()
                tests = " ".join(t_["name"] for t_ in plan["tests"]).lower()
                row.update({"cause_ok": bool(re.search(cause, causes)), "test_ok": not test or bool(re.search(test, tests)), "urgency": plan["urgency"], "urgency_ok": RANK[plan["urgency"]] >= RANK[floor],
                            "causes": [c_["name"] for c_ in plan["causes"]], "tests": [t_["name"] for t_ in plan["tests"]], "medicines": [m["name"] for m in plan["medicines"]],
                            "banned": bool(BANNED.search(re.sub(r"amlodipine", "", json.dumps(plan), flags=re.I) if profile.get("medications") else json.dumps(plan))), "note": plan.get("medicine_note")})
            rows.append(row)
            print(text[:48], "|", {k: row.get(k) for k in ("plan", "cause_ok", "test_ok", "urgency", "urgency_ok", "banned")}, flush=True)
            time.sleep(8)
    planned = [r for r in rows if r["plan"]]
    summary = {k: sum(bool(r.get(k)) for r in planned) for k in ("cause_ok", "test_ok", "urgency_ok", "banned")}
    summary.update({"cases": len(rows), "plans": len(planned)})
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1])
