"""Survey screeners and the diabetes symptom check: registry, adapters and predictions agree."""

import json
from pathlib import Path

import pytest

from workers.common.server import discover

REG = json.loads((Path(__file__).resolve().parents[1] / "registry" / "model_registry.json").read_text(encoding="utf-8"))
NEW = [m for m in REG["models"] if m["id"].startswith(("survey_", "nhanes_")) or m["id"] == "diabetes_symptom_check"]
ADAPTERS = discover("tabular")


def test_every_new_entry_has_a_tested_adapter():
    assert NEW
    for m in NEW:
        assert m["id"] in ADAPTERS and m["integrated"] is True
        met = m["metrics"]
        assert met["sensitivity"] >= 0.75 and met["auc"] >= 0.75, m["id"]  # live bar (user decision 2026-11-10)
        if m["id"] != "diabetes_symptom_check":  # Unverified for another reason: tested only on the same hospital
            assert m["verified"] == (met["sensitivity"] >= 0.80 and met["auc"] >= 0.80), m["id"]  # Verified bar unchanged
        assert set(m["input_schema"]["properties"]) == set(ADAPTERS[m["id"]].input_schema["properties"])


@pytest.mark.parametrize("entry", [m for m in NEW if m["id"].startswith("survey_")], ids=lambda m: m["id"])
def test_survey_risk_rises_with_risk_factors(entry):
    a = ADAPTERS[entry["id"]]()
    a.load()
    low = {
        "age": 25,
        "sex": "Female",
        "smoking": "Never smoked",
        "general_health": "Excellent",
        "height_cm": 160,
        "weight_kg": 55,
        "exercise": "Yes",
        "difficulty_walking": "No",
        "poor_physical_days": 0,
        "poor_mental_days": 0,
    }
    high = {
        "age": 75,
        "sex": "Male",
        "smoking": "Smoke now",
        "general_health": "Poor",
        "height_cm": 165,
        "weight_kg": 100,
        "exercise": "No",
        "difficulty_walking": "Yes",
        "poor_physical_days": 25,
        "poor_mental_days": 25,
    }
    assert not a.validate(low) and not a.validate(high)
    p_low, p_high = a.predict(low), a.predict(high)
    assert p_high.probabilities["Higher risk"] > p_low.probabilities["Higher risk"]
    assert p_low.label == "Lower risk"


def test_diabetes_symptoms():
    a = ADAPTERS["diabetes_symptom_check"]()
    a.load()
    yes = a.predict({"age": 45, "sex": "Male", "passing_urine_often": "Yes", "very_thirsty": "Yes", "sudden_weight_loss": "Yes"})
    no = a.predict({"age": 30, "sex": "Female", "passing_urine_often": "No", "very_thirsty": "No"})
    assert yes.label == "Symptoms fit diabetes" and no.label != yes.label


def test_fatty_liver_and_iron():
    liver = ADAPTERS["nhanes_fatty_liver"]()
    liver.load()
    high = liver.predict(
        {"age": 50, "sex": "Male", "alt": 90, "ast": 60, "waist_cm": 110, "triglycerides": 300, "hba1c": 6.8, "diabetes": "Yes"}
    )
    low = liver.predict(
        {"age": 25, "sex": "Female", "alt": 12, "ast": 18, "waist_cm": 70, "triglycerides": 70, "hba1c": 5.0, "diabetes": "No"}
    )
    assert high.label == "Possible fatty liver" and low.label == "Fatty liver less likely"
    iron = ADAPTERS["nhanes_iron_deficiency"]()
    iron.load()
    assert iron.validate({"age": 40, "sex": "Male", "hemoglobin": 13, "mcv": 85})  # not tested on men: refused
    low_iron = iron.predict({"age": 28, "sex": "Female", "hemoglobin": 9.5, "mcv": 68, "rdw": 18, "mch": 21})
    fine = iron.predict({"age": 28, "sex": "Female", "hemoglobin": 13.5, "mcv": 90, "rdw": 12.5, "mch": 30})
    assert low_iron.label == "Possible low iron stores" and fine.label == "Low iron stores less likely"


def test_new_nhanes_checks():
    bone = ADAPTERS["nhanes_osteoporosis"]()
    bone.load()
    assert bone.validate({"age": 40, "sex": "Female", "weight_kg": 50, "height_cm": 155, "smoking": "Never smoked"})  # under 50
    thin = bone.predict({"age": 78, "sex": "Female", "weight_kg": 45, "height_cm": 150, "smoking": "Smoke now"})
    heavy = bone.predict({"age": 52, "sex": "Male", "weight_kg": 95, "height_cm": 178, "smoking": "Never smoked"})
    assert thin.label == "Higher risk" and heavy.label == "Lower risk"
    ear = ADAPTERS["nhanes_hearing_loss"]()
    ear.load()
    assert ear.validate({"age": 75, "sex": "Male", "hearing_rating": "Good", "hears_whisper": "Yes"})  # outside tested ages
    bad = ear.predict({"age": 65, "sex": "Male", "hearing_rating": "A lot of trouble", "hears_whisper": "No", "hears_normal_voice": "No"})
    good = ear.predict({"age": 30, "sex": "Female", "hearing_rating": "Excellent", "hears_whisper": "Yes", "hears_normal_voice": "Yes"})
    assert bad.label == "Possible hearing loss" and good.label == "Hearing loss less likely"
    kid = ADAPTERS["nhanes_kidney_disease_lab"]()
    kid.load()
    assert kid.predict({"age": 70, "sex": "Male", "diabetes": "Yes", "high_bp": "Yes"}).label == "Higher risk"
