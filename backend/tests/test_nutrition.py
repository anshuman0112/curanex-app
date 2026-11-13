"""AI Nutritionist: food photo -> items + nutrients (totals added in code), non-food refused, targets, meal plan."""
import io
import json
from unittest.mock import MagicMock, patch

from PIL import Image

from api import nutrition


def _png() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(b, format="PNG")
    return b.getvalue()


def _reply(obj) -> MagicMock:
    r = MagicMock(status_code=200)
    r.json.return_value = {"choices": [{"message": {"content": json.dumps(obj)}}]}
    return r


def test_food_photo_totals_are_added_in_code_and_capped(client):
    items = [{"name": "Rice", "portion": "1 cup", "grams": 150, "kcal": 200, "protein_g": 4, "carbs_g": 45, "fat_g": 0.4, "iron_mg": 0.3},
             {"name": "Dal", "portion": "1 katori", "grams": 150, "kcal": 180, "protein_g": 9, "carbs_g": 25, "fat_g": 5, "iron_mg": 2.1, "calcium_mg": 999999}]
    with patch("api.nutrition.httpx.post", return_value=_reply({"is_food": True, "items": items, "confidence": "high"})):
        body = client.post("/v1/food-photo", files={"file": ("m.png", _png(), "image/png")}).json()
    assert [i["name"] for i in body["items"]] == ["Rice", "Dal"]
    assert body["totals"]["kcal"] == 380 and body["totals"]["protein_g"] == 13 and body["totals"]["iron_mg"] == 2.4
    assert body["items"][1]["calcium_mg"] == nutrition.LIMITS["calcium_mg"]  # silly AI numbers are capped


def test_not_food_is_refused(client):
    with patch("api.nutrition.httpx.post", return_value=_reply({"is_food": False})):
        body = client.post("/v1/food-photo", files={"file": ("x.png", _png(), "image/png")}).json()
    assert body["is_food"] is False and body["items"] == []


def test_targets_follow_the_profile():
    man = nutrition.targets({"age": "32", "sex": "Male", "heightCm": "172", "weightKg": "78"})
    woman = nutrition.targets({"age": "32", "sex": "Female", "heightCm": "160", "weightKg": "58"})
    diabetic = nutrition.targets({"age": "32", "sex": "Male", "heightCm": "172", "weightKg": "78", "conditions": ["Type 2 diabetes"]})
    assert woman["iron_mg"] > man["iron_mg"] and woman["kcal"] < man["kcal"]
    assert diabetic["carbs_g"] < man["carbs_g"]


def test_meal_plan_keeps_five_meals_and_strips_doses(client):
    plan = {"meals": [{"time": "Breakfast", "name": "Poha with peanuts", "kcal": 320, "protein": 8, "carbs": 52, "fat": 9, "why": "Light start"}] * 5,
            "tips": ["Take metformin 500 mg daily", "Walk after meals"]}
    with patch("api.nutrition.chat_completion", return_value={"content": json.dumps(plan)}):
        body = client.post("/v1/meal-plan", json={"profile": {"age": "40", "sex": "Male", "diet": "Vegetarian"}}).json()
    assert len(body["meals"]) == 5 and body["error"] is None
    assert "500 mg" not in body["tips"][0]
