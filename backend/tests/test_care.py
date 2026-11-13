"""The AI drafts a care plan, but code decides what reaches the patient."""
import json
from unittest.mock import patch

from api.orchestrator.care import ASKED, care_blocks, is_care_turn, names_rx, safe_medicines, set_urgency
from api.safety.postcheck import scrub
from api.schemas.chat import ChatRequest

PLAN = {"summary": "This may be a muscle strain.", "causes": [{"name": "Muscle strain", "why": "Fits the ache.", "likelihood": "more likely"}, {"name": "Viral illness", "why": "Common cause.", "likelihood": "possible"}],
        "urgency": "doctor_soon", "tests": [{"name": "CBC", "why": "Checks platelets."}],
        "medicines": [{"name": "Paracetamol", "use": "For fever.", "caution": "Avoid alcohol."}, {"name": "Azithromycin", "use": "Antibiotic", "caution": ""}, {"name": "Ibuprofen", "use": "Pain", "caution": ""}],
        "self_care": [{"x": "Rest and drink fluids."}], "see_doctor_if": [{"x": "Fever lasts over 3 days."}]}


def req(message="I have body ache and back pain", **profile):
    return ChatRequest(session_id="s", message=message, context={"profile": profile})


def run(request, plan=PLAN, may_ask=True):
    with patch("api.orchestrator.care.chat_completion", return_value={"content": json.dumps(plan)}), patch("api.orchestrator.care._source", return_value=None):
        return care_blocks(request, may_ask)


def card(blocks):
    return next(b for b in blocks if b["type"] == "care_plan")


def names(plan_card):
    return [m["name"] for m in plan_card["medicines"]]


def test_prescription_medicines_never_reach_the_patient():
    blocks, full = run(req(age=30))
    assert full
    assert names(card(blocks)) == ["Paracetamol", "Ibuprofen"]
    assert "Azithromycin" not in json.dumps(blocks)


def test_a_medicine_with_a_dose_is_dropped():
    plan = {**PLAN, "medicines": [{"name": "Paracetamol", "use": "Take 650 mg for fever", "caution": ""}]}
    assert names(card(run(req(age=30), plan)[0])) == []


def test_profile_removes_unsafe_medicines():
    kept, note = safe_medicines(PLAN["medicines"], req(age=50, conditions=["Hypertension"]))
    assert [m["name"] for m in kept] == ["Paracetamol"] and note
    assert [m["name"] for m in safe_medicines(PLAN["medicines"], req(age=50, allergies=["Paracetamol"]))[0]] == ["Ibuprofen"]
    assert [m["name"] for m in safe_medicines(PLAN["medicines"], req(age=50, conditions=["Fatty liver"]))[0]] == ["Ibuprofen"]


def test_no_medicines_for_children_or_in_pregnancy():
    assert safe_medicines(PLAN["medicines"], req(age=8))[0] == []
    assert safe_medicines(PLAN["medicines"], req("I am pregnant and have fever", age=28))[0] == []
    assert safe_medicines(PLAN["medicines"], req(age=28, pregnant=True))[0] == []


def test_emergency_plan_lists_no_medicines():
    assert names(card(run(req(age=30), {**PLAN, "urgency": "emergency"})[0])) == []


def test_diagnosis_wording_and_unknown_urgency_are_fixed():
    plan = {**PLAN, "urgency": "whenever", "summary": "You have diabetes."}
    blocks, _ = run(req(age=30), plan)
    assert "You have diabetes" not in blocks[0]["text"]
    assert card(blocks)["urgency"] == "doctor_soon"


def test_questions_come_first_then_the_followup_gives_a_plan():
    blocks, full = run(req(age=30), {"questions": [{"q": "How long has this lasted?"}]})
    assert not full and blocks[0]["text"].startswith(ASKED)
    follow = ChatRequest(session_id="s", message="3 days", history=[{"role": "user", "content": "fever"}, {"role": "assistant", "content": blocks[0]["text"]}])
    assert is_care_turn(follow)
    assert run(follow, {**PLAN, "questions": [{"q": "Again?"}]})[1]


def test_a_followup_gets_a_plan_even_if_the_ai_asks_again():
    follow = ChatRequest(session_id="s", message="3 days", history=[{"role": "user", "content": "cough"}, {"role": "assistant", "content": f"{ASKED}, please tell me:"}])
    replies = [{"content": json.dumps({"questions": [{"q": "Again?"}]})}, {"content": json.dumps(PLAN)}]
    with patch("api.orchestrator.care.chat_completion", side_effect=replies), patch("api.orchestrator.care._source", return_value=None):
        blocks, full = care_blocks(follow)
    assert full and card(blocks)["causes"]


def test_prescription_names_are_removed_from_free_text():
    plan = {**PLAN, "self_care": [{"x": "Ask for omeprazole."}, {"x": "Sleep well."}], "summary": "Take azithromycin."}
    blocks, _ = run(req(age=30), plan)
    assert "omeprazole" not in json.dumps(blocks).lower() and "azithromycin" not in json.dumps(blocks).lower()
    assert card(blocks)["self_care"] == ["Sleep well."]


def test_the_patients_own_medicine_may_be_mentioned():
    r = req(age=50, medications=["Amlodipine 5 mg"])
    assert not names_rx("Amlodipine can cause ankle swelling.", r)
    assert names_rx("Consider losartan.", r)


def test_common_symptom_words_reach_the_engine():
    for text in ("yellow eyes, dark urine and no appetite", "big toe very painful at night", "wheezing at night", "I feel unwell"):
        assert is_care_turn(req(text)), text


def test_bad_ai_output_falls_back():
    with patch("api.orchestrator.care.chat_completion", return_value={"content": "not json"}):
        assert care_blocks(req()) is None


def test_general_questions_stay_out_of_the_care_flow():
    assert not is_care_turn(req("What foods have a lot of iron?"))
    assert is_care_turn(req("mujhe bukhar hai"))
    assert is_care_turn(req("मुझे तीन दिन से बुखार है"))


def test_urgency_is_raised_for_serious_causes_and_never_lowered():
    dengue = [{"name": "Dengue fever", "likelihood": "more likely"}]
    assert set_urgency("doctor_soon", dengue, req("fever and body ache", age=30)) == "doctor_today"
    assert set_urgency("self_care", [{"name": "Reflux", "likelihood": "more likely"}], req("burning for 2 weeks", age=30)) == "doctor_soon"
    assert set_urgency("doctor_soon", [{"name": "Flu", "likelihood": "possible"}], req("fever for 4 days", age=30)) == "doctor_today"
    assert set_urgency("self_care", [{"name": "Angina", "likelihood": "less likely"}], req("mild acidity", age=30)) == "self_care"
    assert set_urgency("emergency", dengue, req("fever", age=30)) == "emergency"


def test_ibuprofen_is_removed_with_fever_or_possible_dengue():
    items = [{"name": "Ibuprofen", "use": "Pain"}, {"name": "Paracetamol", "use": "Fever"}]
    assert [m["name"] for m in safe_medicines(items, req("fever and body ache", age=30))[0]] == ["Paracetamol"]
    assert [m["name"] for m in safe_medicines(items, req("body ache", age=30), "Dengue")[0]] == ["Paracetamol"]


def test_nothing_that_could_hide_a_surgical_problem():
    items = [{"name": "Paracetamol", "use": "Pain"}, {"name": "ORS", "use": "Fluids"}]
    kept, note = safe_medicines(items, req("severe tummy pain with vomiting", age=30), "Appendicitis", "doctor_today")
    assert kept == [] and note


def test_medicine_must_match_the_problem():
    items = [{"name": "ORS", "use": "Fluids"}, {"name": "Cetirizine", "use": "Allergy"}]
    assert [m["name"] for m in safe_medicines(items, req("itchy watery eyes and sneezing", age=30), "Allergic rhinitis")[0]] == ["Cetirizine"]


def test_scrub_adds_no_closing_line():
    assert scrub("Rest well.") == "Rest well."


def test_emergency_words_never_reach_the_care_engine(client):
    with patch("api.orchestrator.loop.care_blocks") as care:
        r = client.post("/chat", json={"session_id": "s", "message": "I have severe chest pain"})
    assert r.json()["blocks"][0]["type"] == "escalate"
    care.assert_not_called()


def test_a_denied_fever_does_not_raise_urgency():
    for said in ("mujhe teen din se khansi hai, koi bukhar nahi", "cough for 4 days, no fever", "4 दिन से खांसी, बुखार नहीं है"):
        assert set_urgency("self_care", [], req(said, age=30)) == "self_care", said
    assert set_urgency("self_care", [], req("teen din se bukhar hai", age=30)) == "doctor_today"
    assert set_urgency("self_care", [], req("मुझे बुखार है", age=70)) == "doctor_today"


def test_doctor_style_questions_come_with_tap_answers():
    plan = {"questions": [{"q": "For how many days have you had fever?", "options": ["1-2 days", "3-5 days", "More than 5 days"]},
                          {"q": "Do you feel dizzy or weak?"}]}
    blocks, full = run(req("I have fever"), plan)
    assert not full and blocks[0]["text"].startswith(ASKED)
    qs = blocks[1]["questions"]
    assert blocks[1]["type"] == "questions" and qs[0]["options"] == ["1-2 days", "3-5 days", "More than 5 days"]
    assert qs[1]["options"] == ["Yes", "No"]  # no options given -> simple yes/no
