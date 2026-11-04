"""Diabetes adapter runs on plain numbers, so it is tested directly (no worker needed)."""
import pytest

from workers.tabular.adapters.moizah1_diabetes_prediction import Adapter


@pytest.fixture(scope="module")
def adapter():
    a = Adapter()
    a.load()
    return a


def test_higher_sugar_gives_higher_risk(adapter):
    high = adapter.predict({"glucose": 180, "bmi": 32, "age": 45})
    low = adapter.predict({"glucose": 90, "bmi": 22, "age": 25})
    assert high.label == "Higher risk" and low.label == "Lower risk"
    assert high.probabilities["Higher risk"] > low.probabilities["Higher risk"]
    assert high.probabilities["Higher risk"] >= 0.5 > low.probabilities["Higher risk"]
    assert high.explanation["top_factors"][0][0] == "Blood sugar"


def test_unknown_fields_are_imputed_and_men_are_warned(adapter):
    p = adapter.predict({"glucose": 120, "bmi": 27, "age": 34, "insulin": "", "sex": "Male"})
    assert 0 <= p.probabilities["Higher risk"] <= 1
    assert any("only on women" in w for w in p.warnings)


def test_validation(adapter):
    assert "glucose: missing" in adapter.validate({"bmi": 27, "age": 34})
    assert any("out of range" in e for e in adapter.validate({"glucose": 900, "bmi": 27, "age": 34}))


def test_liver_adapter_flags_abnormal_report():
    from workers.tabular.adapters.yahyazuher_ai_liver_disease_diagnosis_system import Adapter as Liver
    a = Liver()
    a.load()
    high = a.predict({"alt": 180, "ast": 240, "bilirubin": 3.2, "age": 52, "albumin": 2.9, "sex": "Male"})
    low = a.predict({"alt": 22, "ast": 24, "bilirubin": 0.6, "age": 30, "albumin": 4.4, "sex": "Female"})
    assert high.label == "Liver tests need a closer look" and low.label == "Liver tests look less concerning"
    assert high.explanation["top_factors"]


def test_gdm_adapter_normal_sugar_is_not_flagged():
    onnxruntime = pytest.importorskip("onnxruntime")
    assert onnxruntime
    from workers.tf.adapters.srishti_chordia_gestational_diabetes_prediction_using_dl import Adapter as Gdm
    a = Gdm()
    a.load()
    assert a.predict({"fasting_glucose": 150, "age": 29, "bmi": 24}).label == "Higher risk"
    assert a.predict({"fasting_glucose": 80, "age": 29, "bmi": 24}).label == "Lower risk"
    with pytest.raises(ValueError):
        a.predict({"fasting_glucose": 90, "age": 29, "bmi": 24, "sex": "Male"})


def test_heart_adapter_separates_cases_and_warns():
    from workers.tabular.adapters.sneha798_heart_disease_prediction_project import Adapter as Heart
    a = Heart()
    a.load()
    high = a.predict({"age": 62, "sex": "Male", "chest_pain": "Pressure on effort, eases with rest", "pain_on_exertion": "Yes",
                      "resting_bp": 160, "cholesterol": 280})
    low = a.predict({"age": 30, "sex": "Female", "chest_pain": "Chest pain that doesn't feel heart related", "pain_on_exertion": "No",
                     "resting_bp": 110, "cholesterol": 170})
    assert high.label == "Higher risk" and low.label == "Lower risk"
    assert high.probabilities["Higher risk"] >= 0.5 > low.probabilities["Higher risk"]
    assert any("112" in w for w in low.warnings)
    assert a.validate({"age": 40, "sex": "Male", "chest_pain": "bad", "pain_on_exertion": "No"}) == ["chest_pain: not one of the choices"]


def test_thyroid_adapter_reads_tsh():
    from workers.tabular.adapters.cilab_ufersa_thyroid_disease_ai import Adapter as Thyroid
    a = Thyroid()
    a.load()
    assert a.predict({"tsh": 45, "age": 40, "sex": "Female"}).label == "Possible underactive thyroid"
    assert a.predict({"tsh": 1.5, "age": 40, "sex": "Female"}).label == "No clear underactive thyroid pattern"
    assert any("overactive" in w for w in a.predict({"tsh": 0.02}).warnings)
