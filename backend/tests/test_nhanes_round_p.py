"""Replacements for the three Unverified checks (2026-11-14): low muscle mass, heart strain (NT-proBNP), heart muscle injury (troponin).
Each was Verified on a later NHANES cycle (training/train_nhanes_p.py, train_nhanes_q.py)."""
import json
from pathlib import Path

from workers.tabular.adapters.nhanes_heart_damage_troponin import Adapter as Troponin
from workers.tabular.adapters.nhanes_heart_strain_ntprobnp import Adapter as Strain
from workers.tabular.adapters.nhanes_low_muscle_mass import Adapter as Muscle

REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "model_registry.json"
HIGH = {"age": 74, "sex": "Male", "high_bp": "Yes", "heart_attack_or_chd": "Yes", "heart_failure": "Yes", "diabetes": "Yes",
        "stroke": "Yes", "height_cm": 170, "weight_kg": 95}
LOW = {"age": 26, "sex": "Female", "high_bp": "No", "heart_attack_or_chd": "No", "heart_failure": "No", "height_cm": 165, "weight_kg": 58}


def _load(cls):
    a = cls()
    a.load()
    return a


def test_heart_checks_separate_high_and_low_risk():
    for cls, high in ((Strain, "Possible heart strain"), (Troponin, "Possible raised heart troponin")):
        a = _load(cls)
        assert a.validate(HIGH) == [] and a.predict(HIGH).label == high
        assert a.predict(LOW).label != high
    assert any("112" in w for w in _load(Troponin).predict(LOW).warnings)


def test_muscle_check_flags_large_frame_low_muscle_not_thin_people_and_refuses_over_59():
    a = _load(Muscle)
    big = {"age": 45, "sex": "Male", "height_cm": 172, "weight_kg": 118, "waist_cm": 122, "general_health": "Poor", "exercise": "No", "diabetes": "Yes"}
    thin = {"age": 40, "sex": "Male", "height_cm": 180, "weight_kg": 52, "waist_cm": 72}
    assert a.predict(big).label == "Possible low muscle mass" and a.predict(thin).label == "Low muscle mass less likely"
    assert a.validate({"age": 70, "sex": "Male", "height_cm": 170, "weight_kg": 70}) == ["age: this check is for people aged 20 to 59"]


def test_registry_has_the_new_checks_verified_and_the_old_three_retired():
    models = {m["id"]: m for m in json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]}
    for i in ("nhanes_low_muscle_mass", "nhanes_heart_strain_ntprobnp", "nhanes_heart_damage_troponin"):
        m = models[i]
        assert m["integrated"] and m["verified"] and m["metrics"]["sensitivity"] >= 0.80 and m["metrics"]["auc"] >= 0.80
    for i in ("sneha798_heart_disease_prediction_project", "cilab_ufersa_thyroid_disease_ai", "mlmed_torchxrayvision"):
        assert not models[i]["integrated"] and models[i]["retired"]
    live = [m for m in models.values() if m.get("integrated")]
    assert len(live) == 50 and all(m["verified"] for m in live)
