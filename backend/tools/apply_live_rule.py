"""Live rule. 2026-11-11: the user first chose "Verified + near-misses", then asked for every model that met the 0.75 live bar
to be live again. MODE = "all": every model that met the 0.75 bar is live; Verified stays only where an outside test passed 0.80.
MODE = "near_miss": the earlier, stricter rule (kept for reference).

A model is live only if it is Verified, or it was tested on real outside data (another hospital, country or survey year) and reached
sensitivity and AUC >= 0.77 there. Every other model moves to "In testing": still listed with its real scores, never used for
patient results or offered by the AI. Run from backend/ after the add_* scripts (they call it themselves).
"""

import json
from pathlib import Path

MODE = "all"

BACKEND = Path(__file__).resolve().parents[1]
REGISTRIES = [BACKEND / "registry" / "model_registry.json", BACKEND.parent / "docs" / "model_registry.json"]
# Unverified but tested on real outside data at >= 0.77 sensitivity and AUC (fixed on 2026-11-11)
NEAR_MISS = {
    "survey_skin_cancer": "BRFSS 2025: sensitivity 0.842, AUC 0.785",
    "survey_vision_loss": "BRFSS 2025: sensitivity 0.829, AUC 0.777",
    "survey_hearing_difficulty": "BRFSS 2025: sensitivity 0.835, AUC 0.793",
    "survey_high_blood_pressure": "BRFSS 2025: sensitivity 0.829, AUC 0.797",
    "nhanes_low_bone_density": "NHANES 2009-10: sensitivity 0.823, AUC 0.778",
}


# 2026-11-12, user decision: checks with no public outside data are replaced by Verified models and switched off.
RETIRED = {
    "diabetes_symptom_check": "Replaced by the hidden diabetes check (routine blood test), Verified on NHANES 2021-23.",
    "clinic_fetal_ctg": "Replaced by the kidney damage check (routine blood test), Verified on NHANES 2021-23; no outside CTG data with the same inputs exists.",
    "yahyazuher_ai_liver_disease_diagnosis_system": "Replaced by the hepatitis C check (liver function test), Verified on NHANES 2021-23; no outside data with the ILPD inputs and labels passed.",
    "clinic_breast_fna": "Replaced by the kidney function check (urea and uric acid), Verified on NHANES 2021-23; no outside cytology data with the same inputs exists.",
    "srishti_chordia_gestational_diabetes_prediction_using_dl": "Failed its outside test (maternal health dataset, AUC 0.745); replaced by the anaemia check (blood count), Verified on NHANES 2021-23.",
    "clinic_cirrhosis": "Replaced by the fatty liver check from questions (FibroScan truth), Verified on NHANES 2021-23; no outside PBC data with the same inputs exists.",
    "voice_parkinsons_check": "Failed its outside test (Italian recordings, AUC 0.402); replaced by the inflammation check (hs-CRP), Verified on NHANES 2021-23. The voice screen stays as an Unverified demo.",
    "lung_ct_check": "Failed outside its hospital and no third lung CT set exists; replaced by the diabetes check (routine blood test), Verified on NHANES 2021-23.",
    "skin_condition_check": "Failed its outside test (Fitzpatrick17k: average AUC 0.696); replaced by the metabolic syndrome check, Verified on NHANES 2021-23.",
    "oral_condition_check": "No outside mouth-photo set with the same labels exists; replaced by the insulin resistance check, Verified on NHANES 2021-23.",
    "nhanes_hearing_loss": "Missed the 0.80 bar on its outside tests (sensitivity 0.769); replaced by the kidney function risk check (questions), Verified on NHANES 2021-23. The Verified hearing difficulty check covers hearing.",
    "survey_vision_loss": "Missed the 0.80 bar on four outside tests (best AUC 0.798); replaced by the insulin resistance risk check (questions), Verified on NHANES 2021-23.",
    "nhanes_low_bone_density": "Missed the 0.80 bar on every outside test (best AUC 0.791); replaced by the kidney disease check trained in India and Verified in Bangladesh. The Verified osteoporosis check covers bone health.",
    "clinic_glioma_grade": "Replaced by the DME check on OCT, Verified on Duke; no outside data exists with the same gene-test inputs.",
}


def apply() -> tuple[int, int, int]:
    counts = (0, 0, 0)
    for reg in REGISTRIES:
        d = json.loads(reg.read_text(encoding="utf-8"))
        live = verified = testing = 0
        for m in d["models"]:
            if m["id"] in RETIRED:
                m["integrated"], m["retired"] = False, RETIRED[m["id"]]
                m.pop("in_testing", None)
                continue
            if m.get("in_testing"):
                m["integrated"] = True  # re-evaluate below
            if not m.get("integrated"):
                continue
            if MODE == "all" or m.get("verified") or m["id"] in NEAR_MISS:
                m.pop("in_testing", None)
                live += 1
                verified += bool(m.get("verified"))
                if m["id"] in NEAR_MISS:
                    m["near_miss"] = NEAR_MISS[m["id"]]
            else:
                m["integrated"] = False
                m["in_testing"] = True
                testing += 1
        reg.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        counts = (live, verified, testing)
    return counts


if __name__ == "__main__":
    live, verified, testing = apply()
    print(f"live {live} (Verified {verified}, near-miss {live - verified}), in testing {testing}")
