"""Workout planner + gym coach: safety filtering and anti-hallucination checks (no network: the LLM is faked)."""
import json
from pathlib import Path

from api import workout
from api.routers import gym

ROOT = Path(__file__).resolve().parents[2]


def fake_llm(content: str):
    return lambda *a, **k: {"content": content}


def test_catalog_copies_identical():
    app = (ROOT / "src" / "data" / "exercises.json").read_text(encoding="utf-8")
    api = (ROOT / "backend" / "api" / "data" / "exercises.json").read_text(encoding="utf-8")
    assert app == api, "run: python backend/tools/build_exercise_catalog.py"


def test_catalog_is_well_formed():
    cats = set(workout.CATALOG["categories"])
    tags = set(workout.CATALOG["conditionLabels"])
    assert len(workout.EX) >= 60
    for e in workout.EX.values():
        assert e["category"] in cats, e["id"]
        assert set(e["avoid"]) <= tags and set(e["caution"]) <= tags, e["id"]
        assert e["steps"], e["id"]
        if e["mode"] == "reps":
            assert e.get("count"), e["id"]
        if e["mode"] == "hold":
            assert e.get("hold", {}).get("targets"), e["id"]
    assert {e["category"] for e in workout.EX.values()} == cats


def test_health_tags_and_bp_block():
    tags, notes, blocked = workout.health_tags({"age": "70", "conditions": ["Knee osteoarthritis", "High BP"], "heightCm": "160", "weightKg": "95"}, None)
    assert {"knee", "hypertension", "senior", "obesity_severe"} <= set(tags) and blocked is None
    _, _, blocked = workout.health_tags({"conditions": []}, {"bp_sys": 185, "bp_dia": 100})
    assert blocked and "doctor" in blocked
    tags, notes, _ = workout.health_tags({"conditions": []}, {"bp_sys": 165, "bp_dia": 95, "hr_resting": 105})
    assert "hypertension" in tags and "heart" in tags and notes


def test_allowed_excludes_unsafe_and_missing_equipment():
    pool = workout.allowed(["knee", "pregnancy"], [], "advanced")
    for e in pool.values():
        assert not {"knee", "pregnancy"} & set(e["avoid"])
        assert e["equipment"] == "none"
    assert "lunge" not in pool and "burpee" not in pool and "dumbbell-curl" not in pool


def test_rule_plan_without_llm(monkeypatch):
    monkeypatch.setattr(workout, "chat_completion", lambda *a, **k: None)
    for goal in workout.GOALS:
        p = workout.build_plan({"age": "30", "conditions": []}, None, {"goal": goal, "days": 4, "minutes": 30, "equipment": ["mat", "dumbbells", "chair", "wall"]})
        assert p["source"] == "rules" and len(p["days"]) == 7
        assert sum(not d["rest"] for d in p["days"]) == 4
        for d in p["days"]:
            for it in d["items"]:
                assert it["id"] in workout.EX


def test_ai_plan_hallucinations_are_removed(monkeypatch):
    reply = {"days": [
        {"day": "Mon", "rest": False, "title": "Legs", "items": [
            {"id": "squat", "sets": 99, "reps": 500},          # out of range -> clamped
            {"id": "flying-dragon-kick", "sets": 3, "reps": 10},  # invented -> dropped
            {"id": "burpee", "sets": 3, "reps": 10},            # unsafe for knee -> dropped
            {"id": "plank", "sets": 2, "seconds": 600}]},
        {"day": "Tue", "rest": True, "items": []},
        {"day": "Wed", "rest": False, "title": "Yoga", "items": [{"id": "tadasana", "sets": 1, "seconds": 30}]},
    ], "tips": ["Drink water."]}
    monkeypatch.setattr(workout, "chat_completion", fake_llm(json.dumps(reply)))
    p = workout.build_plan({"conditions": ["knee pain", "hypertension"]}, None, {"goal": "strength", "days": 3, "minutes": 30, "equipment": ["mat"]})
    assert p["source"] == "ai"
    mon = p["days"][0]["items"]
    ids = [i["id"] for i in mon]
    assert "flying-dragon-kick" not in ids and "burpee" not in ids
    sq = next(i for i in mon if i["id"] == "squat")
    assert sq["sets"] <= 2 and sq["reps"] <= workout.EX["squat"]["range"]["reps"][1] and sq.get("caution")  # knee caution caps sets
    pl = next(i for i in mon if i["id"] == "plank")
    assert pl["seconds"] <= 30  # hypertension caps holds
    assert sum(not d["rest"] for d in p["days"]) == 3  # missing training day filled by rules
    assert any(x["id"] == "burpee" for x in p["excluded"])


def test_blocked_plan_has_no_exercises(monkeypatch):
    monkeypatch.setattr(workout, "chat_completion", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM must not be called")))
    p = workout.build_plan({}, {"bp_sys": 190, "bp_dia": 115}, {})
    assert p["blocked"] and p["days"] == []


def test_coach_rejects_drift(monkeypatch):
    monkeypatch.setattr(gym, "_tts", lambda t: None)
    monkeypatch.setattr(gym, "chat_completion", fake_llm("Great squat! Now drop down and do some burpees and a forward lunge right away."))
    out = gym._coach(gym.CoachIn(event="ongoing_form_check", exercise="Bodyweight squat", exercise_id="squat", issue="not deep", cue="Go lower."))
    assert out["source"] == "fallback" and out["text"] == "Go lower."
    monkeypatch.setattr(gym, "chat_completion", fake_llm("Sit your hips back and go a little lower."))
    out = gym._coach(gym.CoachIn(event="ongoing_form_check", exercise="Bodyweight squat", exercise_id="squat", issue="not deep", cue="Go lower."))
    assert out["source"] == "llm"
