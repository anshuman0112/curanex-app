"""Doctor module API: locked without the doctor key; answers keep only library-supported, cited statements."""

from unittest.mock import patch

import pytest

PASSAGES = [
    {"id": "p1", "source": "StatPearls", "title": "Community-Acquired Pneumonia", "section": "Evaluation", "url": "u1", "licence": "CC BY-NC-ND 4.0",
     "part": 1, "text": "Chest radiography is required to confirm the diagnosis of community-acquired pneumonia in patients with cough, fever and dyspnea. "
                        "A complete blood count and pulse oximetry should be obtained."},
    {"id": "p2", "source": "StatPearls", "title": "Tuberculosis", "section": "Evaluation", "url": "u2", "licence": "CC BY-NC-ND 4.0",
     "part": 1, "text": "Sputum smear microscopy and nucleic acid amplification testing are used when cough lasts more than two weeks."},
]
CASE = {"patient": {"age": 45, "sex": "male"}, "symptoms": "fever for 3 days, a lot of cough, difficulty breathing"}


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("DOCTOR_DEV_KEY", "test-doctor-key")
    return {"X-Doctor-Key": "test-doctor-key"}


def test_locked_without_key(client, monkeypatch):
    monkeypatch.delenv("DOCTOR_DEV_KEY", raising=False)
    assert client.post("/doctor/tests", json=CASE).status_code == 403


def test_wrong_key_is_refused(client, key):
    assert client.post("/doctor/tests", json=CASE, headers={"X-Doctor-Key": "patient-guess"}).status_code == 403


def test_tests_keep_only_supported_cited_items(client, key):
    reply = {"tests": [
        {"test": "Chest X-ray", "text": "Chest radiography is required to confirm community-acquired pneumonia with cough, fever and dyspnea.", "cite": ["p1"]},
        {"test": "Pulse oximetry", "text": "Complete blood count and pulse oximetry should be obtained.", "cite": ["p1"]},
        {"test": "CT brain", "text": "A CT scan of the brain is always needed.", "cite": ["p1"]},
        {"test": "Made-up test", "text": "Serum unicorn levels.", "cite": ["p9"]},
    ]}
    with patch("api.knowledge.answer.search", return_value=PASSAGES), patch("api.knowledge.answer.ask_json", return_value=reply):
        r = client.post("/doctor/tests", json=CASE, headers=key)
    body = r.json()
    assert r.status_code == 200
    assert [t["test"] for t in body["tests"]] == ["Chest X-ray", "Pulse oximetry"]
    assert body["dropped"] == 2
    assert body["tests"][0]["sources"][0]["title"] == "Community-Acquired Pneumonia"


def test_unsupported_diagnosis_goes_to_also_consider(client, key):
    replies = iter([{"candidates": ["Community-acquired pneumonia", "Dengue"]},
                    {"evidence": [{"text": "Cough, fever and dyspnea fit community-acquired pneumonia.", "cite": ["p1"]}]},
                    {"evidence": [{"text": "Dengue causes cough.", "cite": ["p1"]}]}])
    with patch("api.knowledge.answer.search", return_value=PASSAGES[:1]), patch("api.knowledge.answer.ask_json", side_effect=lambda *a, **k: next(replies)):
        body = client.post("/doctor/differential", json=CASE, headers=key).json()
    assert [d["disease"] for d in body["supported"]] == ["Community-acquired pneumonia"]
    assert body["also_consider"] == ["Dengue"]


def test_treatment_asks_for_missing_details_first(client, key):
    body = client.post("/doctor/treatment", json={"case": CASE, "disease": "Community-acquired pneumonia"}, headers=key).json()
    assert body["medicines"] == [] and "weight_kg" in body["missing_info"] and "allergies" in body["missing_info"]


def test_emergency_rules_still_fire(client, key):
    case = {**CASE, "symptoms": "crushing chest pain and cannot breathe"}
    with patch("api.knowledge.answer.search", return_value=[]):
        body = client.post("/doctor/tests", json=case, headers=key).json()
    assert body["emergency"]
