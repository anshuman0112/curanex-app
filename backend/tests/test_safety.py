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
