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
    assert high.explanation["top_factors"][0][0] == "Blood sugar"


def test_unknown_fields_are_imputed_and_men_are_warned(adapter):
    p = adapter.predict({"glucose": 120, "bmi": 27, "age": 34, "insulin": "", "sex": "Male"})
    assert 0 <= p.probabilities["Higher risk"] <= 1
    assert any("only on women" in w for w in p.warnings)


def test_validation(adapter):
    assert "glucose: missing" in adapter.validate({"bmi": 27, "age": 34})
    assert any("out of range" in e for e in adapter.validate({"glucose": 900, "bmi": 27, "age": 34}))
