from api.safety import post_check, red_flag


def test_flags():
    assert red_flag("crushing chest pain spreading to my left arm") == "Possible heart attack"
    assert red_flag("my face is drooping and slurred speech") == "Possible stroke"
    assert red_flag("I can't breathe") == "Severe breathing difficulty"
    assert red_flag("I want to end my life") == "Thoughts of self-harm"

def test_no_false_alarm():
    assert red_flag("my chest muscles hurt after gym") is None
    assert red_flag("what's a healthy breakfast") is None

def test_post_check():
    out = post_check("Take 500 mg paracetamol.")
    assert "500 mg" not in out and "doctor" in out.lower()

def test_post_check_ml_only_for_medicines():
    assert "5 ml" not in post_check("Give 5 ml of the cough syrup twice a day.")
    assert "250 ml of water" in post_check("Drink 250 ml of water with breakfast.")
    assert "**" not in post_check("**Oats** are good. See a doctor if unsure.")


def test_red_team_set_all_escalate():
    from tools.verify_e2e import RED_TEAM
    assert [t for t in RED_TEAM if not red_flag(t)] == []


def test_everyday_messages_do_not_escalate():
    for text in ("What are the benefits of walking?", "my left arm is sore after cricket", "I have a mild headache", "my shoes fit badly",
                 "what is a normal heart rate", "I have a chest cold and a cough", "How much water should I drink?",
                 "is zero sugar cola ok", "food in the freezer"):
        assert red_flag(text) is None, text


def test_app_and_backend_rules_match():
    import re
    from pathlib import Path

    from api.safety.rules import RED_FLAGS
    source = Path(__file__).resolve().parents[2] / "src" / "lib" / "safety.ts"
    if not source.exists():  # Docker image has no app source
        return
    assert re.findall(r"\[/(.+)/i, '", source.read_text(encoding="utf-8")) == [p for p, _ in RED_FLAGS]


def test_lab_values_are_not_treated_as_doses():
    assert "142 mg/dL" in post_check("Fasting sugar is 142 mg/dL, above the range. Ask your doctor.")
    assert "40 IU/L" in post_check("ALT is 40 IU/L. Ask your doctor.")
    assert "5 mg/kg" not in post_check("Give 5 mg/kg daily.")
    assert "500 mg" not in post_check("Take 500 mg paracetamol.")
