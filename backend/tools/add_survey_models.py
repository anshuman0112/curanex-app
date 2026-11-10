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
LIVE_BAR = 0.75  # user decision 2026-11-10: on at sensitivity and AUC >= 0.75 on unseen data
VERIFIED_BAR = 0.80  # Verified still needs 0.80


def is_live(test: dict) -> bool:
    return test["sensitivity"] >= LIVE_BAR and test["auc"] >= LIVE_BAR


def is_verified(test: dict) -> bool:
    return test["sensitivity"] >= VERIFIED_BAR and test["auc"] >= VERIFIED_BAR


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
        "Heart attack risk score from lifestyle questions (age, smoking, weight, exercise, diabetes). Am I at risk of a heart attack, coronary heart disease risk, smoker's heart attack risk.",
        "An ECG, blood pressure, cholesterol and sugar tests, and a doctor's visit.",
    ),
    "stroke": (
        "Neurology",
        "Stroke risk",
        "Stroke risk score from lifestyle questions (age, smoking, weight, exercise, BP, diabetes). Paralysis risk, brain attack risk, chance of a stroke in future.",
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
        "Chronic kidney disease risk score from lifestyle questions. Are my kidneys at risk, kidney failure risk, kidney problems with diabetes or BP.",
        "A creatinine/eGFR blood test and a urine albumin test.",
    ),
    "arthritis": (
        "Musculoskeletal",
        "Arthritis risk",
        "Arthritis risk score from lifestyle questions. Stiff joints every morning, aching joints for months, rheumatoid arthritis or gout worry, joints in many places.",
        "A doctor's examination; sometimes X-rays or blood tests.",
    ),
    "depression": (
        "Mental health",
        "Depression risk",
        "Depression risk score from questions. Feeling low or sad for weeks, no interest in things I used to enjoy, hopeless, mental health check.",
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
        "Type 2 diabetes risk score from lifestyle questions only, for someone who has never had a sugar test. Overweight, no exercise, smoker, could I get diabetes.",
        "A fasting glucose or HbA1c blood test.",
    ),
    "cancer": (
        "Oncology",
        "Cancer risk (general)",
        "General cancer risk score from lifestyle questions (age, smoking, alcohol, weight). Am I at risk of cancer, smoking and cancer risk.",
        "Age-appropriate screening advised by a doctor.",
    ),
    "high_blood_pressure": (
        "Cardiovascular",
        "High blood pressure risk",
        "Hypertension risk score from lifestyle questions: could I have hypertension, chance that I have high BP without knowing.",
        "Measuring your blood pressure: a pharmacy or home BP machine is enough to start.",
    ),
    "skin_cancer": (
        "Cancer",
        "Skin cancer risk (not melanoma)",
        "Skin cancer risk score from lifestyle questions. Basal or squamous cell skin cancer risk, lots of sun exposure, sunburn and skin cancer.",
        "A doctor or dermatologist checking any new, growing or bleeding skin spot. The US survey is mostly fair-skinned people, "
        "so it likely overstates the risk for darker Indian skin.",
    ),
    "vision_loss": (
        "Eye",
        "Serious vision problems risk",
        "Risk of serious difficulty seeing even with glasses, from lifestyle questions. Eyesight getting worse, low vision risk with diabetes.",
        "An eye check-up with vision and eye pressure tests; people with diabetes need a yearly retina check.",
    ),
    "memory_problems": (
        "Neurology",
        "Memory and concentration problems risk",
        "Risk of serious trouble remembering, concentrating or making decisions, from lifestyle questions. Forgetfulness, memory getting worse, brain fog, early dementia worry.",
        "A doctor's memory assessment; thyroid, vitamin B12 and sugar tests; and checking for low mood, poor sleep or medicine side effects, which are common and treatable causes.",
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
        "verified": is_verified(test),
        "description": desc,
        "status_note": "Survey-based screener tested on the next survey year (different people), not on Indian data."
        + ("" if is_verified(test) else " Unverified: between 0.75 and 0.80 on that test."),
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
        "description": "Diabetes symptom check. Passing urine often, very thirsty all the time, always hungry, sudden weight loss, blurred vision, wounds healing slowly.",
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
        "description": "Fatty liver (NAFLD) check from liver enzymes and waist size. Is my liver fatty, fatty liver grade, belly fat and liver, high triglycerides and liver fat.",
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
    "osteoporosis": {
        "id": "nhanes_osteoporosis",
        "categories": ["Musculoskeletal"],
        "task": "Osteoporosis",
        "name": "Osteoporosis risk check",
        "labels": ["Lower risk", "Higher risk"],
        "who": "aged 50 and over",
        "truth": "hip bone density measured by DXA scan (femoral neck T-score -2.5 or lower)",
        "next": "A DXA bone density scan confirms it. Calcium, vitamin D, exercise and not smoking protect bones.",
        "description": "Osteoporosis and weak bones risk. Thin bones, bone density, fracture after a small fall, menopause and bones, getting shorter, bone loss in old age.",
    },
    "kidney_disease_lab": {
        "id": "nhanes_kidney_disease_lab",
        "categories": ["Renal"],
        "task": "Hidden kidney disease",
        "name": "Hidden kidney disease risk",
        "labels": ["Lower risk", "Higher risk"],
        "who": "aged 20 and over",
        "truth": "kidney damage measured by blood and urine tests (eGFR below 60 or urine albumin/creatinine 30 mg/g or more)",
        "next": "A creatinine (eGFR) blood test and a urine albumin test confirm it.",
        "description": "Silent kidney disease that has not been diagnosed yet. Should I get my kidneys tested, kidney check with diabetes or high BP, protein in urine.",
    },
    "low_bone_density": {
        "id": "nhanes_low_bone_density",
        "categories": ["Musculoskeletal"],
        "task": "Low bone density (osteopenia)",
        "name": "Low bone density check",
        "labels": ["Lower risk", "Higher risk"],
        "who": "aged 50 and over",
        "truth": "hip bone density measured by DXA scan (femoral neck T-score -1 or lower)",
        "next": "A DXA bone density scan confirms it. Calcium, vitamin D and weight-bearing exercise help keep bones strong.",
        "description": "Osteopenia, early bone thinning before osteoporosis. Bone density a bit low, weak bones after menopause, should I get a bone density scan.",
    },
    "gout": {
        "id": "nhanes_gout",
        "categories": ["Musculoskeletal"],
        "task": "Gout",
        "name": "Gout risk check",
        "labels": ["Lower risk", "Higher risk"],
        "who": "aged 20 and over",
        "truth": "a doctor's diagnosis of gout",
        "next": "A uric acid blood test and a doctor's examination of a painful joint confirm it.",
        "description": "Gout risk. Sudden very painful swollen big toe, uric acid, painful joint at night, gout attack worry.",
    },
    "heart_failure": {
        "id": "nhanes_heart_failure",
        "categories": ["Cardiovascular"],
        "task": "Heart failure",
        "name": "Heart failure risk check",
        "labels": ["Lower risk", "Higher risk"],
        "who": "aged 20 and over",
        "truth": "a doctor's diagnosis of congestive heart failure",
        "next": "An echocardiogram and a BNP blood test confirm it. Breathlessness when lying flat or swollen ankles need a doctor soon.",
        "description": "Heart failure risk. Weak heart, breathless when lying down, swollen ankles and legs, heart pumping weakly, heart failure after a heart attack.",
    },
    "hearing_loss": {
        "id": "nhanes_hearing_loss",
        "categories": ["ENT"],
        "task": "Hearing loss",
        "name": "Hearing loss check",
        "labels": ["Hearing loss less likely", "Possible hearing loss"],
        "who": "aged 20 to 69",
        "truth": "hearing measured in a sound booth (better ear average above 25 dB)",
        "next": "A hearing test (audiometry) with an ENT doctor or audiologist confirms it.",
        "description": "Hearing loss check. Can't hear well, people seem to mumble, TV too loud, ringing in the ears, hard to hear a whisper, deaf in one ear.",
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
        "verified": is_verified(test),
        "description": info["description"],
        "status_note": "Tested on a later NHANES cycle (different people and years), with the condition measured, not self-reported."
        + ("" if is_verified(test) else " Unverified: between 0.75 and 0.80 on that test."),
    }


CLINIC = {
    "fetal_ctg": {
        "id": "clinic_fetal_ctg",
        "categories": ["Obstetrics"],
        "task": "Fetal distress on CTG",
        "name": "CTG fetal distress check (clinic)",
        "labels": ["CTG pattern looks normal", "Suspicious or abnormal CTG pattern"],
        "trained_on": "{n} CTG recordings from a hospital in Portugal (UCI Cardiotocography, Ayres de Campos et al.), each classed normal, suspect or pathological by three obstetricians",
        "limitations": "Clinic-grade: needs values read from a CTG machine, so it is for doctors and midwives, not patients. Not a diagnosis. Tested only on recordings from the same hospital. It flagged about {fp}% of normal CTGs. Reduced baby movements, bleeding or pain in pregnancy need urgent care.",
        "credit": "Trained by the Curanex team on the UCI Cardiotocography dataset (Ayres de Campos, Bernardes, Garrido, Marques de Sá, Pereira-Leite)",
        "license": "CC BY 4.0 (UCI)",
        "description": "For doctors and midwives: CTG (cardiotocography) values in pregnancy, fetal heart rate variability, decelerations, accelerations, is the CTG suspicious or pathological.",
    },
    "breast_fna": {
        "id": "clinic_breast_fna",
        "categories": ["Cancer"],
        "task": "Breast lump cytology (FNA)",
        "name": "Breast FNA cytology check (clinic)",
        "labels": ["Cytology looks benign", "Cytology suggests malignancy"],
        "trained_on": "{n} breast fine-needle aspirates scored by cytologists at the University of Wisconsin hospitals, with confirmed outcomes (UCI Breast Cancer Wisconsin, Original)",
        "limitations": "Clinic-grade: needs the cytology scores a pathologist gives, so it is for doctors, not patients. Not a diagnosis. Tested only on samples from the same hospitals (1990s). It flagged about {fp}% of benign samples. A biopsy decides.",
        "credit": "Trained by the Curanex team on the UCI Breast Cancer Wisconsin (Original) dataset (Wolberg, Mangasarian)",
        "license": "CC BY 4.0 (UCI)",
        "description": "For doctors: breast lump fine-needle aspiration (FNA) cytology scores, clump thickness, cell uniformity, bare nuclei, is the breast FNA malignant.",
    },
}


CLINIC.update(
    {
        "mammo_mass": {
            "id": "clinic_mammo_mass",
            "categories": ["Cancer"],
            "task": "Breast mass on mammogram",
            "name": "Mammogram mass check (clinic)",
            "labels": ["Findings fit a benign mass", "Findings fit a malignant mass"],
            "trained_on": "{n} mammogram masses from Erlangen, Germany, with biopsy-confirmed outcomes (UCI Mammographic Mass)",
            "limitations": "Clinic-grade: needs the radiologist's BI-RADS, shape, margin and density, so it is for doctors. Not a diagnosis. "
            "Tested only on masses from the same centre. It flagged about {fp}% of benign masses. A biopsy decides.",
            "credit": "Trained by the Curanex team on the UCI Mammographic Mass dataset (Elter, Schulz-Wendtland, Wittenberg)",
            "license": "CC BY 4.0 (UCI)",
            "description": "For doctors: mammogram mass findings, BI-RADS, mass shape, margin and density, is the breast mass malignant.",
        },
        "cirrhosis": {
            "id": "clinic_cirrhosis",
            "categories": ["Hepatology"],
            "task": "Liver cirrhosis",
            "name": "Liver cirrhosis check (clinic)",
            "labels": ["Cirrhosis less likely", "Possible cirrhosis"],
            "trained_on": "{n} patients with primary biliary cholangitis in a Mayo Clinic trial, USA, with liver biopsy stage (UCI Cirrhosis)",
            "limitations": "Clinic-grade: for people already known to have chronic liver disease, with blood tests and an examination. Not a diagnosis. "
            "Trained on one liver disease (primary biliary cholangitis) in the 1970s-80s and tested only on held-out patients from the same trial. "
            "It flagged about {fp}% of patients without cirrhosis. A FibroScan, ultrasound or biopsy confirms it.",
            "credit": "Trained by the Curanex team on the UCI Cirrhosis Patient Survival dataset (Mayo Clinic, Dickson et al.)",
            "license": "CC BY 4.0 (UCI)",
            "description": "For doctors: chronic liver disease patient, does the liver have cirrhosis, low platelets, low albumin, high bilirubin, ascites.",
        },
        "glioma_grade": {
            "id": "clinic_glioma_grade",
            "categories": ["Cancer"],
            "task": "Glioma grade",
            "name": "Glioma grade check (clinic)",
            "labels": ["Profile fits lower-grade glioma", "Profile fits glioblastoma (grade 4)"],
            "trained_on": "{n} brain tumour patients from The Cancer Genome Atlas (TCGA LGG and GBM), with gene test results (UCI Glioma Grading)",
            "limitations": "Clinic-grade: needs tumour gene test results, so it is for neuro-oncology teams. Not a diagnosis. "
            "Tested only on held-out TCGA patients. It flagged about {fp}% of lower-grade gliomas. The pathologist's WHO grade decides.",
            "credit": "Trained by the Curanex team on the UCI Glioma Grading Clinical and Mutation Features dataset (TCGA)",
            "license": "CC BY 4.0 (UCI)",
            "description": "For doctors: glioma tumour gene results IDH1, ATRX, PTEN, EGFR, TP53, is the brain glioma low grade or glioblastoma.",
        },
    }
)


def clinic_entry(name: str) -> dict:
    from workers.tabular.clinic import schema_for as clinic_schema

    meta = json.loads((BACKEND / "workers/tabular/models/clinic" / f"{name}.json").read_text(encoding="utf-8"))
    info, test = CLINIC[name], meta["metrics"]["test"]
    fp = round(100 * (1 - test["specificity"]))
    return {
        "id": info["id"],
        "repo": "UCI Machine Learning Repository",
        "url": None,
        "categories": info["categories"],
        "tasks": [info["task"]],
        "input": "Clinic values",
        "modality": "tabular",
        "status": "W",
        "app_tier": "C",
        "tier": "C",
        "wave": "replacement",
        "name": info["name"],
        "worker": "tabular",
        "adapter": f"workers/tabular/adapters/{info['id']}.py",
        "conditions": [info["task"]],
        "input_schema": clinic_schema(name, labels=True),
        "output_labels": info["labels"],
        "trained_on": info["trained_on"].format(n=meta["training_rows"]),
        "limitations": info["limitations"].format(fp=fp),
        "license": info["license"],
        "credit": info["credit"],
        "metrics": {
            "accuracy": test["accuracy"],
            "sensitivity": test["sensitivity"],
            "specificity": test["specificity"],
            "auc": test["auc"],
            "eval_set": f"Held-out split, same source (n={test['n']}, {test['positives']} positive), scored once",
            "measured": "2026-11-10",
        },
        "integrated": True,
        "verified": False,
        "description": info["description"],
        "status_note": "Unverified: tested only on held-out data from the same source. Clinic-grade inputs.",
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
    for name in CLINIC:
        meta = json.loads((BACKEND / "workers/tabular/models/clinic" / f"{name}.json").read_text(encoding="utf-8"))
        if meta["live"]:
            entries.append(clinic_entry(name))
            print("on:", name, meta["metrics"]["test"])
    for target in NHANES:
        meta = json.loads((BACKEND / "workers/tabular/models/nhanes" / f"{target}.json").read_text(encoding="utf-8"))
        if is_live(meta["metrics"]["test"]):
            entries.append(nhanes_entry(target))
            print("on:", target, meta["metrics"]["test"])
    for path in sorted(SURVEY.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        adapter = ADAPTERS / f"survey_{meta['target']}.py"
        if not is_live(meta["metrics"]["test_2023"]):
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
