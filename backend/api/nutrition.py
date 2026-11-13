"""AI Nutritionist: food photo -> dishes, portions and nutrients; daily targets; personal Indian meal plan.

Food photos use the free Groq vision model (api.vision.VISION_MODEL); the meal plan uses the main text model. Every number is an
AI ESTIMATE from the photo (portion sizes are guessed), labelled as such in the app. Totals are added up in code, not by the AI.
Daily targets: Mifflin-St Jeor energy (light activity) and ICMR-NIN 2020 recommended amounts for Indian adults (rounded).
"""
import base64
import io
import json
import re
import time

import httpx
from PIL import Image

from api.clients.llm import chat_completion
from api.core.config import get_settings
from api.core.logging import log
from api.safety.postcheck import scrub
from api.vision import VISION_MODEL

NUTRIENTS = {  # key: (label, unit)
    "kcal": ("Calories", "kcal"), "protein_g": ("Protein", "g"), "carbs_g": ("Carbohydrates", "g"), "fat_g": ("Fat", "g"),
    "fibre_g": ("Fibre", "g"), "sugar_g": ("Sugar", "g"), "iron_mg": ("Iron", "mg"), "calcium_mg": ("Calcium", "mg"),
    "vitamin_c_mg": ("Vitamin C", "mg"), "vitamin_a_ug": ("Vitamin A", "µg"), "vitamin_b12_ug": ("Vitamin B12", "µg"),
    "vitamin_d_ug": ("Vitamin D", "µg"), "folate_ug": ("Folate", "µg"), "sodium_mg": ("Sodium", "mg"), "potassium_mg": ("Potassium", "mg"),
}
LIMITS = {"kcal": 3000, "protein_g": 150, "carbs_g": 400, "fat_g": 200, "fibre_g": 60, "sugar_g": 200, "iron_mg": 60, "calcium_mg": 2500,
          "vitamin_c_mg": 1000, "vitamin_a_ug": 5000, "vitamin_b12_ug": 50, "vitamin_d_ug": 100, "folate_ug": 2000, "sodium_mg": 6000, "potassium_mg": 6000}

FOOD_PROMPT = ("You are a nutritionist who knows Indian and international food. Look at this photo. If it is not food or drink, reply "
               '{"is_food": false}. Otherwise identify each dish or item, estimate the portion visible (in grams and in household terms such as '
               '"1 katori", "2 rotis", "1 cup") and estimate its nutrients for THAT portion using standard food composition values (IFCT 2017 / USDA). '
               "Reply only JSON: {\"is_food\": true, \"items\": [{\"name\": \"...\", \"portion\": \"...\", \"grams\": number, "
               + ", ".join(f'"{k}": number' for k in NUTRIENTS) + "}], \"confidence\": \"high|medium|low\"}")

PLAN_PROMPT = """You are an Indian clinical nutritionist. Make a one-day meal plan for this person. Respect their diet type strictly
(Vegetarian: no meat, fish or egg; Eggetarian: egg allowed; Vegan: no dairy, egg or meat; Jain: also no onion, garlic, potato or other root vegetables),
their allergies, and their health conditions (for example diabetes: low glycaemic, no added sugar; high blood pressure: low salt; kidney disease: moderate protein,
low potassium and salt; anaemia: iron-rich foods with vitamin C; pregnancy: folate, iron and calcium). Use common, affordable Indian home food.
Aim close to these daily targets: {targets}.
Reply only JSON: {{"meals": [{{"time": "Breakfast|Mid-morning|Lunch|Evening snack|Dinner", "name": "dishes with portions", "kcal": number,
"protein": number, "carbs": number, "fat": number, "why": "one short reason it suits them"}}], "tips": ["up to 4 short tips for their conditions"]}}
Exactly 5 meals. No medicines, no supplements doses."""


def _num(v, cap: float) -> float:
    try:
        x = float(re.sub(r"[^0-9.]", "", str(v)) or 0)
    except ValueError:
        return 0.0
    return round(min(max(x, 0.0), cap), 1)


def _jpeg(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1024, 1024))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def analyse_food(data: bytes) -> dict:
    """{"is_food", "items": [{name, portion, grams, <nutrients>}], "totals", "confidence"}; {"error"} if the AI is unavailable."""
    s = get_settings()
    base, key = s.llm_endpoint
    body = {"model": VISION_MODEL, "temperature": 0, "max_tokens": 2000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": FOOD_PROMPT}, {"type": "image_url", "image_url": {"url": _jpeg(data)}}]}]}
    try:
        for attempt in range(5):
            r = httpx.post(f"{base.rstrip('/')}/chat/completions", timeout=90, headers={"Authorization": f"Bearer {key}"}, json=body)
            if r.status_code != 429 or attempt == 4:
                break
            time.sleep(min(float(r.headers.get("retry-after") or 2 ** attempt), 20))
        r.raise_for_status()
        found = re.findall(r"\{.*\}", r.json()["choices"][0]["message"]["content"] or "", re.S)
        raw = json.loads(found[-1]) if found else {}
    except Exception as e:
        log.warning("food_photo_error", error=type(e).__name__, detail=str(e)[:200])
        return {"error": "The food reader is busy. Please try again in a minute."}
    if not raw.get("is_food"):
        return {"is_food": False, "items": [], "totals": {}}
    items = []
    for it in (raw.get("items") or [])[:10]:
        if not isinstance(it, dict) or not str(it.get("name") or "").strip():
            continue
        items.append({"name": scrub(str(it["name"]))[:80], "portion": scrub(str(it.get("portion") or ""))[:60], "grams": _num(it.get("grams"), 2000),
                      **{k: _num(it.get(k), LIMITS[k]) for k in NUTRIENTS}})
    totals = {k: round(sum(i[k] for i in items), 1) for k in NUTRIENTS}
    return {"is_food": bool(items), "items": items, "totals": totals, "confidence": str(raw.get("confidence") or "medium")[:10],
            "note": "AI estimate from the photo. Portions are guessed, so treat the numbers as approximate."}


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def targets(profile: dict) -> dict:
    """Daily targets for this person. Energy: Mifflin-St Jeor x 1.4 (light activity), adjusted for the goal; others: ICMR-NIN 2020."""
    age, h, w = _f(profile.get("age")) or 30, _f(profile.get("heightCm")) or 165, _f(profile.get("weightKg")) or 65
    female = str(profile.get("sex") or "").lower().startswith("f")
    bmr = 10 * w + 6.25 * h - 5 * age + (-161 if female else 5)
    kcal = bmr * 1.4
    goal = str(profile.get("goal") or "")
    if goal == "Lose weight":
        kcal -= 400
    elif goal == "Build muscle":
        kcal += 250
    kcal = round(max(1200, min(kcal, 3500)) / 50) * 50
    conditions = " ".join(profile.get("conditions") or []).lower()
    protein = round(w * (1.2 if goal == "Build muscle" else 0.83 if "kidney" in conditions else 1.0))
    if "diabet" in conditions:  # lower-carb split for diabetes (about 45% of energy), more from fat and protein
        carbs = round(kcal * 0.45 / 4)
        fat = round((kcal - protein * 4 - carbs * 4) / 9)
    else:
        fat = round(kcal * 0.27 / 9)
        carbs = round((kcal - protein * 4 - fat * 9) / 4)
    return {"kcal": kcal, "protein_g": protein, "carbs_g": carbs, "fat_g": fat, "fibre_g": 30, "iron_mg": 29 if female else 19,
            "calcium_mg": 1000, "vitamin_c_mg": 65 if female else 80, "vitamin_a_ug": 840 if female else 1000, "vitamin_b12_ug": 2.5,
            "sodium_mg": 2000, "source": "Energy: Mifflin-St Jeor, light activity. Nutrients: ICMR-NIN 2020 recommended amounts for Indian adults."}


def meal_plan(profile: dict) -> dict:
    t = targets(profile)
    person = {k: profile.get(k) for k in ("age", "sex", "heightCm", "weightKg", "diet", "goal", "conditions", "allergies")}
    want = f"{t['kcal']} kcal, protein {t['protein_g']} g, carbs {t['carbs_g']} g, fat {t['fat_g']} g, fibre 30 g, iron {t['iron_mg']} mg"
    msg = chat_completion([{"role": "system", "content": PLAN_PROMPT.format(targets=want)},
                           {"role": "user", "content": f"<person>{json.dumps(person, ensure_ascii=False)}</person>"}], json_mode=True, max_tokens=1800)
    try:
        raw = json.loads((msg or {}).get("content") or "")
    except json.JSONDecodeError:
        raw = {}
    meals = []
    for m in (raw.get("meals") or [])[:6]:
        if isinstance(m, dict) and m.get("name"):
            meals.append({"time": str(m.get("time") or "")[:20], "name": scrub(str(m["name"]))[:300], "why": scrub(str(m.get("why") or ""))[:160],
                          **{k: int(_num(m.get(k), 2000)) for k in ("kcal", "protein", "carbs", "fat")}})
    return {"targets": t, "meals": meals, "tips": [scrub(str(x))[:200] for x in (raw.get("tips") or [])[:4]],
            "error": None if meals else "The meal planner is busy. Please try again in a minute."}
