"""Switches on the survey risk screeners that passed their test (training/train_brfss.py), plus the diabetes symptom check.

For each passing model: writes a small adapter in workers/tabular/adapters/ and adds or updates its registry entry in
backend/registry/model_registry.json and docs/model_registry.json (kept identical). Failing models are left out.
Run from backend/.
"""

import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from workers.tabular.survey import BASE_SCHEMA, CONDITION_LABEL, REQUIRED, schema_for  # noqa: E402

SURVEY = BACKEND / "workers" / "tabular" / "models" / "survey"
ADAPTERS = BACKEND / "workers" / "tabular" / "adapters"
REGISTRIES = [BACKEND / "registry" / "model_registry.json", BACKEND.parent / "docs" / "model_registry.json"]
LABELS = {
    "age": "Age",
    "sex": "Sex",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "smoking": "Do you smoke?",
    "chewed_tobacco": "Do you use gutka, khaini, paan with tobacco or other chewed tobacco?",
    "heavy_drinking": "Do you drink more than 14 drinks a week (men) or 7 (women)?",
    "exercise": "Did you exercise or play sport in the last month?",
    "general_health": "How is your health in general?",
    "poor_physical_days": "Days in the last month your physical health was not good",
    "poor_mental_days": "Days in the last month your mental health was not good (stress, low mood)",
    "difficulty_walking": "Do you have serious difficulty walking or climbing stairs?",
    "difficulty_concentrating": "Do you have serious difficulty concentrating, remembering or deciding?",
} | {f"has_{c}": f"Has a doctor told you that you have {v}?" for c, v in CONDITION_LABEL.items()}
UNITS = {"height_cm": "cm", "weight_kg": "kg", "poor_physical_days": "days", "poor_mental_days": "days"}
PROFILE = {"age", "sex", "height_cm", "weight_kg"}
# Asked first after the required questions (the form shows five).
ORDER = [
    "difficulty_walking",
    "exercise",
    "chewed_tobacco",
    "heavy_drinking",
    "poor_physical_days",
    "poor_mental_days",
    "difficulty_concentrating",
]
INFO = {
    "heart_disease": (
        "Cardiovascular",
        "Coronary heart disease risk",
        "Heart disease risk from everyday questions. Worried about my heart, heart attack risk, chest tightness, family history of heart disease, smoker heart risk, how healthy is my heart.",
        "An ECG, blood pressure, cholesterol and sugar tests, and a doctor's visit.",
    ),
    "stroke": (
        "Neurology",
        "Stroke risk",
        "Stroke risk from everyday questions. Worried about a stroke, brain attack, paralysis risk, family history of stroke, high BP and stroke.",
        "Blood pressure, sugar and cholesterol checks. Sudden weakness, face drooping or slurred speech is an emergency: call 112.",
    ),
    "copd": (
        "Respiratory",
        "COPD risk",
        "COPD and chronic lung disease risk. Smoker's cough, breathless on stairs, long-term cough with phlegm, chronic bronchitis, emphysema, lungs damaged by smoking or chulha smoke.",
        "A spirometry (lung function) test and a doctor's visit.",
    ),
    "kidney_disease": (
        "Renal",
        "Chronic kidney disease risk",
        "Kidney disease risk. Worried about my kidneys, diabetes or BP and kidney damage, swelling of feet, foamy urine.",
        "A creatinine/eGFR blood test and a urine albumin test.",
    ),
    "arthritis": (
        "Musculoskeletal",
        "Arthritis risk",
        "Arthritis risk. Joint pain, stiff joints in the morning, knee pain, swollen joints, gout, rheumatoid arthritis worry.",
        "A doctor's examination; sometimes X-rays or blood tests.",
    ),
    "depression": (
        "Mental health",
        "Depression risk",
        "Depression risk. Feeling low or sad for weeks, no interest in things, stress, can't concentrate, mental health check.",
        "Talking to a doctor or counsellor. If you have thoughts of harming yourself, call Tele-MANAS 14416 now.",
    ),
    "asthma": (
        "Respiratory",
        "Asthma risk",
        "Asthma risk. Wheezing, chest tightness, cough at night, breathless with dust or cold air, inhaler.",
        "A spirometry test and a doctor's visit.",
    ),
    "diabetes": (
        "Endocrine",
        "Type 2 diabetes risk",
        "Diabetes risk from lifestyle questions, without a blood test. Worried about sugar, family history of diabetes, overweight and diabetes risk.",
        "A fasting glucose or HbA1c blood test.",
    ),
    "cancer": (
        "Oncology",
        "Cancer risk (general)",
        "General cancer risk from lifestyle questions. Worried about cancer, smoking and cancer risk.",
        "Age-appropriate screening advised by a doctor.",
    ),
    "high_blood_pressure": (
        "Cardiovascular",
        "High blood pressure risk",
        "High blood pressure risk. Worried about BP, hypertension, headache and BP, family history of high BP.",
        "Measuring your blood pressure: a pharmacy or home BP machine is enough to start.",
    ),
    "high_cholesterol": (
        "Cardiovascular",
        "High cholesterol risk",
        "High cholesterol risk. Worried about cholesterol, lipid profile, fatty food and heart.",
        "A lipid profile blood test.",
    ),
}


def schema(target: str) -> dict:
    s = schema_for(target)
    props = s["properties"]
    keys = REQUIRED + [k for k in ORDER if k in props] + [k for k in props if k not in REQUIRED and k not in ORDER]
    out = {}
    for k in keys:
        spec = dict(props[k]) | {"label": LABELS[k]}
        if k in UNITS:
            spec["unit"] = UNITS[k]
        if k in PROFILE:
            spec["from_profile"] = True
        out[k] = spec
    return {"type": "object", "required": REQUIRED, "properties": out}


def survey_entry(meta: dict) -> dict:
    t = meta["target"]
    cat, task, desc, next_step = INFO[t]
    test = meta["metrics"]["test_2023"]
    year = 2021 if "2021" in meta["trained_on"] else 2022
    return {
        "id": f"survey_{t}",
        "repo": "CDC BRFSS (public survey data)",
        "url": "https://www.cdc.gov/brfss/",
        "categories": [cat],
        "tasks": [task],
        "input": "Questions you can answer",
        "modality": "tabular",
        "status": "W",
        "app_tier": "A",
        "tier": "A",
        "wave": "replacement",
        "name": f"{meta['title']} check",
        "worker": "tabular",
        "adapter": f"workers/tabular/adapters/survey_{t}.py",
        "conditions": [task],
        "input_schema": schema(t),
        "output_labels": ["Lower risk", "Higher risk"],
        "trained_on": f"{re.search(r'survey, ([0-9,]+ adults)', meta['trained_on']).group(1)} who answered the US CDC BRFSS health survey in {year}, "
        "with whether a doctor had ever told them they have this condition",
        "limitations": "Not a diagnosis. It estimates how likely someone with your answers is to have been diagnosed, from a US survey; "
        "it has not been tested on Indian people, and it can't see blood tests, scans or symptoms you didn't mention. "
        f"In testing it flagged about {round(100 * (1 - test['specificity']))}% of people without the condition. "
        f"Scores are scaled so 50% is the flagging point. Next step: {next_step}",
        "license": "Public domain (US CDC)",
        "credit": "Trained by the Curanex team on the US CDC Behavioral Risk Factor Surveillance System (BRFSS) survey",
        "metrics": {
            "accuracy": None,
            "sensitivity": test["sensitivity"],
            "specificity": test["specificity"],
            "auc": test["auc"],
            "eval_set": f"BRFSS 2023, {test['n']:,} different people ({test['positives']:,} with the condition), scored once",
            "measured": "2026-11-10",
        },
        "integrated": True,
        "verified": True,
        "description": desc,
        "status_note": "Survey-based screener. Verified on the next year's survey (different people), not on Indian data.",
    }


def symptom_entry() -> dict:
    meta = json.loads((BACKEND / "workers/tabular/models/diabetes_symptom_check/model_meta.json").read_text(encoding="utf-8"))
    test = meta["metrics"]["test"]
    from workers.tabular.adapters.diabetes_symptom_check import SYMPTOMS, Adapter

    labels = {
        "passing_urine_often": "Passing urine much more often than usual?",
        "very_thirsty": "Very thirsty all the time?",
        "sudden_weight_loss": "Lost weight suddenly without trying?",
        "weakness": "Feeling weak?",
        "very_hungry": "Feeling very hungry all the time?",
        "genital_thrush": "Repeated genital itching or yeast infection?",
        "blurred_vision": "Blurred vision?",
        "itching": "Itching skin?",
        "irritability": "Getting irritated easily?",
        "slow_healing": "Cuts or wounds healing slowly?",
        "muscle_weakness": "Weakness in an arm or leg?",
        "muscle_stiffness": "Muscle stiffness?",
        "hair_loss": "Hair loss?",
        "obesity": "Are you overweight?",
    }
    props = {
        "age": {"type": "integer", "minimum": 16, "maximum": 100, "label": "Age", "from_profile": True},
        "sex": {"type": "string", "label": "Sex", "from_profile": True},
    }
    props |= {k: {"type": "string", "enum": ["Yes", "No"], "label": labels[k]} for k in SYMPTOMS}
    assert set(props) == set(Adapter.input_schema["properties"])
    return {
        "id": "diabetes_symptom_check",
        "repo": "UCI Early stage diabetes risk prediction",
        "url": "https://archive.ics.uci.edu/dataset/529",
        "categories": ["Endocrine"],
        "tasks": ["Diabetes symptom check"],
        "input": "Questions you can answer",
        "modality": "tabular",
        "status": "W",
        "app_tier": "A",
        "tier": "A",
        "wave": "replacement",
        "name": "Diabetes symptom check",
        "worker": "tabular",
        "adapter": "workers/tabular/adapters/diabetes_symptom_check.py",
        "conditions": ["Diabetes (from symptoms)"],
        "input_schema": {"type": "object", "required": Adapter.input_schema["required"], "properties": props},
        "output_labels": ["Unlikely", "Likely"],
        "trained_on": "312 people at Sylhet Diabetes Hospital, Bangladesh, newly diabetic or not, with their symptoms (UCI, Islam et al. 2020)",
        "limitations": "Not a diagnosis. Only symptoms are used, and many people with diabetes have none, so a low result doesn't rule it out. "
        "Tested only on people from the same hospital (104 held out). A fasting glucose or HbA1c blood test is the real check.",
        "license": "CC BY 4.0 (UCI)",
        "credit": "Trained by the Curanex team on the UCI Early stage diabetes risk prediction dataset (Islam, Ferdousi, Rahman, Bushra)",
        "metrics": {
            "accuracy": test["accuracy"],
            "sensitivity": test["sensitivity"],
            "specificity": test["specificity"],
            "auc": test["auc"],
            "eval_set": f"Held-out split, same hospital (n={test['n']}), scored once",
            "measured": "2026-11-10",
        },
        "integrated": True,
        "verified": False,
        "description": "Diabetes symptom check. Passing urine often, very thirsty, always hungry, sudden weight loss, blurred vision, wounds healing slowly, itching, tired and weak, worried I have sugar.",
        "status_note": "Unverified: tested only on the same hospital's patients.",
    }


NHANES = {
    "fatty_liver": {
        "id": "nhanes_fatty_liver",
        "categories": ["Hepatology"],
        "task": "Fatty liver (NAFLD)",
        "name": "Fatty liver check (blood tests)",
        "labels": ["Fatty liver less likely", "Possible fatty liver"],
        "who": "aged 18 and over",
        "truth": "liver fat measured by FibroScan (CAP 285 dB/m or more)",
        "next": "An ultrasound or FibroScan confirms it.",
        "description": "Fatty liver check from a liver function test. High SGPT or SGOT, ALT AST raised, fatty liver worry, NAFLD, belly fat and liver, high triglycerides.",
    },
    "iron_deficiency": {
        "id": "nhanes_iron_deficiency",
        "categories": ["Hematology"],
        "task": "Iron deficiency",
        "name": "Iron deficiency check (blood count)",
        "labels": ["Low iron stores less likely", "Possible low iron stores"],
        "who": "aged 12 and over",
        "truth": "iron stores measured by ferritin (below 15 ng/mL)",
        "next": "A serum ferritin test confirms it. Tested only on women and girls aged 12 to 49, so it is only offered to them.",
        "description": "Iron deficiency check from a CBC (complete blood count). Low haemoglobin, anaemia, low MCV, high RDW, tired and pale, heavy periods, low iron.",
    },
}


def nhanes_entry(target: str) -> dict:
    from workers.tabular.nhanes import schema_for as nhanes_schema

    meta = json.loads((BACKEND / "workers/tabular/models/nhanes" / f"{target}.json").read_text(encoding="utf-8"))
    info, test = NHANES[target], meta["metrics"]["test"]
    people = re.search(r"([0-9,]+) people", meta["trained_on"]).group(1)
    years = re.search(r"NHANES ([0-9-]+)", meta["trained_on"]).group(1)
    return {
        "id": info["id"],
        "repo": "CDC NHANES (public survey data)",
        "url": "https://www.cdc.gov/nchs/nhanes/",
        "categories": info["categories"],
        "tasks": [info["task"]],
        "input": "Lab report values",
        "modality": "tabular",
        "status": "W",
        "app_tier": "A",
        "tier": "A",
        "wave": "replacement",
        "name": info["name"],
        "worker": "tabular",
        "adapter": f"workers/tabular/adapters/{info['id']}.py",
        "conditions": [info["task"]],
        "input_schema": nhanes_schema(target, labels=True),
        "output_labels": info["labels"],
        "trained_on": f"{people} people {info['who']} in the US CDC NHANES survey ({years}), with {info['truth']}",
        "limitations": "Not a diagnosis. Trained and tested on US survey participants, not on Indian patients. "
        f"In testing it flagged about {round(100 * (1 - test['specificity']))}% of people without the condition. "
        f"Scores are scaled so 50% is the flagging point. {info['next']}",
        "license": "Public domain (US CDC)",
        "credit": "Trained by the Curanex team on the US CDC National Health and Nutrition Examination Survey (NHANES)",
        "metrics": {
            "accuracy": None,
            "sensitivity": test["sensitivity"],
            "specificity": test["specificity"],
            "auc": test["auc"],
            "eval_set": f"{meta['test_set']}: {test['n']:,} people ({test['positives']:,} with the condition), scored once",
            "measured": "2026-11-10",
        },
        "integrated": True,
        "verified": True,
        "description": info["description"],
        "status_note": "Verified on a later NHANES cycle (different people and years), with the condition measured, not self-reported.",
    }


ADAPTER = '''"""Survey risk screener: {title}.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""
from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_{target}"
    target = "{target}"
    input_schema = schema_for("{target}")
'''


def main() -> None:
    entries = [symptom_entry()]
    for target in NHANES:
        meta = json.loads((BACKEND / "workers/tabular/models/nhanes" / f"{target}.json").read_text(encoding="utf-8"))
        if meta["passed"]:
            entries.append(nhanes_entry(target))
            print("on:", target, meta["metrics"]["test"])
    for path in sorted(SURVEY.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        adapter = ADAPTERS / f"survey_{meta['target']}.py"
        if not meta["passed"]:
            adapter.unlink(missing_ok=True)
            print("left off (failed):", meta["target"], meta["metrics"]["test_2023"])
            continue
        adapter.write_text(ADAPTER.format(title=meta["title"], target=meta["target"]), encoding="utf-8")
        entries.append(survey_entry(meta))
        print("on:", meta["target"], meta["metrics"]["test_2023"])
    ids = {e["id"] for e in entries}
    for reg in REGISTRIES:
        data = json.loads(reg.read_text(encoding="utf-8"))
        data["models"] = [
            m for m in data["models"] if m["id"] not in ids and not (m["id"].startswith(("survey_", "nhanes_")) and m["id"] not in ids)
        ] + entries
        reg.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(entries)} entries written; BASE_SCHEMA has {len(BASE_SCHEMA)} questions")


if __name__ == "__main__":
    main()
