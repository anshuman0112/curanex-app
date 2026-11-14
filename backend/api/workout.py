"""Personalised weekly workout plan (gym, yoga, cardio/weight loss, mobility, low-impact, breathing).

Anti-hallucination design:
  1. The exercise catalog (api/data/exercises.json, same file as the app's src/data/exercises.json) is the only
     source of exercises, safety tags and ranges. The LLM never invents an exercise.
  2. Safety is deterministic and runs BEFORE the LLM: condition keywords, age, BMI and smartwatch BP/HR map to
     tags; exercises whose `avoid` list hits a tag are removed. A very high BP reading blocks the plan entirely.
  3. The LLM only chooses ids and order from the allowed list. Every id, set, rep and hold time it returns is
     checked against the catalog and clamped; anything unknown or unsafe is dropped.
  4. If the LLM is unavailable or returns nothing usable, a rule-based planner builds the plan (source="rules").
"""
import json
from pathlib import Path

from api.clients.llm import chat_completion
from api.core.logging import log
from api.safety.postcheck import scrub

CATALOG = json.loads((Path(__file__).parent / "data" / "exercises.json").read_text(encoding="utf-8"))
EX: dict[str, dict] = {e["id"]: e for e in CATALOG["exercises"]}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LEVELS = {"beginner": 0, "intermediate": 1, "advanced": 2}
GOALS = ("weight_loss", "strength", "yoga", "flexibility", "general", "low_impact")
TRAIN_DAYS = {2: [0, 3], 3: [0, 2, 4], 4: [0, 1, 3, 4], 5: [0, 1, 2, 4, 5], 6: [0, 1, 2, 3, 4, 5], 7: list(range(7))}
REST_SECONDS = 30
# Which catalog categories each goal may use (keeps a "yoga" plan yoga). Warm-up/cool-down come from mobility/breathing/yoga.
GOAL_CATEGORIES = {
    "weight_loss": {"cardio", "bodyweight", "core", "low_impact", "strength", "mobility", "yoga", "breathing"},
    "strength": {"strength", "bodyweight", "core", "mobility", "yoga", "breathing"},
    "yoga": {"yoga", "breathing", "mobility"},
    "flexibility": {"yoga", "mobility", "breathing"},
    "general": {"strength", "bodyweight", "core", "cardio", "low_impact", "mobility", "yoga", "breathing"},
    "low_impact": {"low_impact", "mobility", "yoga", "breathing", "core"},
}
GROWABLE = {"strength", "bodyweight", "core", "cardio", "low_impact", "yoga"}
NOT_ADVICE = "This plan is general fitness guidance, not medical advice. Stop if you feel pain, dizziness, chest pain or breathlessness."


def _num(v, default: float = 0.0) -> float:
    try:
        return float(str(v).strip() or default)
    except (TypeError, ValueError):
        return default


def health_tags(profile: dict, watch: dict | None) -> tuple[list[str], list[str], str | None]:
    """(tags, notes, blocked). Deterministic: keywords from the catalog's condition table, age, BMI, BP, resting HR."""
    text = " | ".join(str(c) for c in (profile.get("conditions") or [])).lower()
    tags = [t for t, words in CATALOG["conditions"].items() if any(w in text for w in words)]
    notes: list[str] = []
    if _num(profile.get("age")) >= 65:
        tags.append("senior")
    h, w = _num(profile.get("heightCm")) / 100, _num(profile.get("weightKg"))
    if h > 0 and w > 0 and w / (h * h) >= 35:
        tags.append("obesity_severe")
    watch = watch or {}
    sys_bp, dia_bp, rhr = _num(watch.get("bp_sys")), _num(watch.get("bp_dia")), _num(watch.get("hr_resting"))
    if sys_bp >= 180 or dia_bp >= 110:
        return tags, notes, (f"Your latest blood pressure reading ({int(sys_bp)}/{int(dia_bp)}) is very high. "
                             "Please don't start a workout plan until a doctor has checked it. If you have chest pain, a severe headache or trouble breathing, use SOS.")
    if sys_bp >= 160 or dia_bp >= 100:
        tags.append("hypertension")
        notes.append(f"Your recent blood pressure ({int(sys_bp)}/{int(dia_bp)}) is high, so this plan keeps the intensity low. Check with your doctor.")
    if rhr >= 100:
        tags.append("heart")
        notes.append(f"Your resting heart rate ({int(rhr)} bpm) is high, so high-intensity work is left out. Please mention it to your doctor.")
    if "heart" in tags:
        notes.append("With a heart condition, get your doctor's OK before starting and keep the effort light to moderate.")
    if "pregnancy" in tags:
        notes.append("During pregnancy, follow your obstetrician's advice. Lying-on-the-belly, deep core and high-impact moves are left out.")
    if "hypertension" in tags:
        notes.append("Breathe steadily throughout: never hold your breath. Holds are kept to 30 seconds.")
    return sorted(set(tags)), notes, None


def allowed(tags: list[str], equipment: list[str], level: str) -> dict[str, dict]:
    have = set(equipment or []) | {"none"}
    max_level = LEVELS.get(level, 0)
    if "senior" in tags or "heart" in tags:
        max_level = min(max_level, 1)
    return {i: e for i, e in EX.items()
            if not set(e["avoid"]) & set(tags) and e["equipment"] in have and LEVELS[e["level"]] <= max_level}


def _clamp_item(e: dict, sets, reps, seconds, tags: list[str]) -> dict:
    r = e["range"]
    lo, hi = r["sets"]
    s = int(min(max(round(_num(sets, e["defaults"]["sets"])), lo), hi))
    caution = [CATALOG["conditionLabels"].get(t, t) for t in e["caution"] if t in tags]
    if caution:
        s = min(s, 2)
    item = {"id": e["id"], "name": e["name"], "mode": e["mode"], "category": e["category"], "sets": s}
    if e["mode"] == "reps":
        lo, hi = r["reps"]
        n = int(min(max(round(_num(reps, e["defaults"]["reps"])), lo), hi))
        if "senior" in tags:
            n = min(n, 15)
        item["reps"] = n
    else:
        lo, hi = r["seconds"]
        n = int(min(max(round(_num(seconds, e["defaults"]["seconds"])), lo), hi))
        if e["mode"] == "hold" and "hypertension" in tags:
            n = min(n, 30)
        item["seconds"] = n
    if caution:
        item["caution"] = caution
    return item


def item_seconds(it: dict) -> int:
    work = it["reps"] * 3 if it["mode"] == "reps" else it["seconds"]
    return it["sets"] * work + (it["sets"] - 1) * REST_SECONDS + 20  # +20 s to move to the next exercise


def _fit(items: list[dict], minutes: int) -> list[dict]:
    """Trim from the middle (keep warm-up first and cool-down last) until the day fits ~minutes."""
    limit = minutes * 60 * 1.15
    items = items[:]
    while len(items) > 3 and sum(item_seconds(i) for i in items) > limit:
        items.pop(-2)
    return items


def _fill(items: list[dict], minutes: int, pool: dict[str, dict], tags: list[str]) -> list[dict]:
    """Day too short: first add sets (within catalog range, not for caution items), then add exercises of the same kinds."""
    target = minutes * 60 * 0.85

    def total() -> int:
        return sum(item_seconds(i) for i in items)

    main = items[1:-1] if len(items) > 2 else items
    grew = True
    while total() < target and grew:
        grew = False
        for it in main:
            if total() >= target:
                break
            if it["category"] in GROWABLE and not (it["category"] == "yoga" and it["mode"] == "timed") and not it.get("caution") and it["sets"] < min(EX[it["id"]]["range"]["sets"][1], 4):
                it["sets"] += 1
                grew = True
    cats = {i["category"] for i in main} or {"bodyweight"}
    used = {i["id"] for i in items}
    extra = sorted((e for e in pool.values() if e["category"] in cats and e["id"] not in used), key=lambda e: (LEVELS[e["level"]], e["id"]))
    while total() < target and extra and len(items) < 9:
        items.insert(max(1, len(items) - 1), _clamp_item(extra.pop(0), None, None, None, tags))
    return items


def _clean(text, n: int) -> str:
    text = str(text).translate({0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: ", ", 0x2019: "'", 0x201C: '"', 0x201D: '"'})
    return scrub(text)[:n]


# ---------- rule-based planner (fallback and baseline) ----------
LOWER = {"quads", "glutes", "hamstrings", "inner thighs", "hips", "legs", "hip flexors"}
UPPER = {"chest", "triceps", "shoulders", "biceps", "back", "forearms", "arms"}
MUSCLES = {"Lower body": LOWER, "Lower body strength": LOWER, "Upper body": UPPER, "Arms + shoulders": {"biceps", "triceps", "shoulders", "forearms"},
           "Upper body + core": UPPER | {"core", "abs", "obliques"}}
THEMES: dict[str, list[tuple[str, list[str]]]] = {
    "weight_loss": [("Cardio + core", ["cardio", "core"]), ("Lower body strength", ["bodyweight", "strength"]), ("Yoga recovery", ["yoga"]),
                    ("Cardio intervals", ["cardio", "low_impact"]), ("Upper body + core", ["strength", "bodyweight", "core"]), ("Full body burn", ["cardio", "bodyweight"]),
                    ("Mobility + breathing", ["mobility", "yoga"])],
    "strength": [("Lower body", ["bodyweight", "strength"]), ("Upper body", ["strength", "bodyweight"]), ("Core + mobility", ["core", "mobility"]),
                 ("Full body strength", ["strength", "bodyweight"]), ("Arms + shoulders", ["strength"]), ("Yoga stretch", ["yoga"]), ("Core", ["core"])],
    "yoga": [("Standing poses", ["yoga"]), ("Balance + strength", ["yoga"]), ("Gentle flow + breathing", ["yoga", "breathing"]),
             ("Hips + hamstrings", ["yoga", "mobility"]), ("Strength yoga", ["yoga"]), ("Restorative", ["yoga", "breathing"]), ("Flow", ["yoga"])],
    "flexibility": [("Mobility", ["mobility", "yoga"]), ("Yoga stretch", ["yoga"]), ("Posture", ["yoga", "mobility"]),
                    ("Hamstrings + hips", ["yoga", "mobility"]), ("Gentle flow", ["yoga"]), ("Restorative", ["yoga", "breathing"]), ("Mobility", ["mobility"])],
    "general": [("Full body", ["bodyweight", "strength"]), ("Cardio", ["cardio", "low_impact"]), ("Yoga", ["yoga"]),
                ("Strength + core", ["strength", "core"]), ("Cardio + mobility", ["cardio", "mobility"]), ("Yoga + breathing", ["yoga", "breathing"]), ("Core", ["core"])],
    "low_impact": [("Gentle strength", ["low_impact"]), ("Chair + mobility", ["low_impact", "mobility"]), ("Gentle yoga", ["yoga", "breathing"]),
                   ("Walking + balance", ["low_impact"]), ("Mobility", ["mobility", "yoga"]), ("Breathing + stretch", ["breathing", "yoga"]), ("Gentle strength", ["low_impact"])],
}


def _rule_day(theme: tuple[str, list[str]], pool: dict[str, dict], tags: list[str], minutes: int, seed: int) -> dict:
    title, cats = theme
    warm = [e for e in pool.values() if e["category"] == "mobility" and e["mode"] == "timed"]
    cool = [e for e in pool.values() if e["category"] in ("breathing", "yoga") and e["mode"] in ("hold", "timed") and e["level"] == "beginner"]
    main = [e for e in pool.values() if e["category"] in cats and e not in warm and e not in cool]
    if title in MUSCLES:
        focused = [e for e in main if set(e["muscles"]) & MUSCLES[title]]
        main = focused or main
    main = sorted(main, key=lambda e: (LEVELS[e["level"]], e["id"]))
    if main:
        k = seed % len(main)
        main = main[k:] + main[:k]
    items = []
    if warm:
        items.append(_clamp_item(warm[seed % len(warm)], 1, None, 60, tags))
    items += [_clamp_item(e, None, None, None, tags) for e in main[:6]]
    if cool:
        items.append(_clamp_item(cool[seed % len(cool)], 1, None, 60 if cool[seed % len(cool)]["mode"] == "hold" else 120, tags))
    return {"title": title, "focus": ", ".join(CATALOG["categories"][c] for c in cats), "items": _fill(_fit(items, minutes), minutes, pool, tags), "why": ""}


def rule_plan(goal: str, days: int, minutes: int, pool: dict[str, dict], tags: list[str]) -> list[dict]:
    themes = THEMES.get(goal, THEMES["general"])
    train = TRAIN_DAYS[days]
    out = []
    for d in range(7):
        if d in train:
            n = train.index(d)
            out.append({"day": DAYS[d], "rest": False, **_rule_day(themes[n % len(themes)], pool, tags, minutes, seed=d * 3 + n)})
        else:
            out.append({"day": DAYS[d], "rest": True, "title": "Rest / light walk", "focus": "Recovery", "items": [], "why": "Rest days let your body recover."})
    return out


# ---------- AI planner ----------
SYSTEM = """You are a certified fitness coach building a 7-day plan inside a health app.
Rules you must follow:
- Use ONLY exercise ids from <allowed>. Never invent exercises, ids, numbers, diagnoses or medical claims.
- Exactly {days} training days; the other days are rest days with an empty items list. Spread hard days apart.
- Each training day: start with a warm-up (mobility or low-impact timed item), 4-6 main exercises matching the day's focus, end with a cool-down (yoga hold or breathing).
- Fit about {minutes} minutes per training day. Respect the goal "{goal}" and level "{level}".
- Items flagged caution: use at most 2 sets.
- "why" = one short sentence linking the day to the person's goal/health details (no medical advice).
Return JSON only:
{{"days":[{{"day":"Mon","rest":false,"title":"short title","focus":"short focus","why":"one sentence","items":[{{"id":"<allowed id>","sets":3,"reps":12}},{{"id":"<allowed id>","sets":2,"seconds":30}}]}}, ... 7 days Mon..Sun],"tips":["max 3 short tips"]}}
Use "reps" for mode=reps items and "seconds" for mode=hold/timed items."""


def ai_plan(person: dict, prefs: dict, pool: dict[str, dict], tags: list[str]) -> tuple[list[dict] | None, list[str]]:
    lines = [f'{e["id"]} | {e["name"]} | {e["category"]} | mode={e["mode"]} | {e["level"]} | impact={e["impact"]} | {",".join(e["muscles"])}'
             + (" | caution" if set(e["caution"]) & set(tags) else "") for e in pool.values()]
    msg = chat_completion([
        {"role": "system", "content": SYSTEM.format(days=prefs["days"], minutes=prefs["minutes"], goal=prefs["goal"], level=prefs["level"])},
        {"role": "user", "content": f"<person>{json.dumps(person, ensure_ascii=False)}</person>\n<health_tags>{', '.join(tags) or 'none'}</health_tags>\n<allowed>\n" + "\n".join(lines) + "\n</allowed>"},
    ], json_mode=True, max_tokens=3500)
    try:
        raw = json.loads((msg or {}).get("content") or "")
    except json.JSONDecodeError:
        return None, []
    days_raw = {str(d.get("day", ""))[:3].title(): d for d in (raw.get("days") or []) if isinstance(d, dict)}
    out, dropped = [], 0
    for name in DAYS:
        d = days_raw.get(name) or {"rest": True}
        items = []
        for it in (d.get("items") or [])[:9]:
            e = pool.get(str((it or {}).get("id", "")))
            if not e:  # unknown or unsafe id from the model: drop it
                dropped += 1
                continue
            if any(x["id"] == e["id"] for x in items):
                continue
            items.append(_clamp_item(e, it.get("sets"), it.get("reps"), it.get("seconds"), tags))
        rest = bool(d.get("rest")) or not items
        out.append({"day": name, "rest": rest, "title": _clean(d.get("title") or ("Rest" if rest else "Workout"), 40),
                    "focus": _clean(d.get("focus") or "", 60), "why": _clean(d.get("why") or "", 180),
                    "items": [] if rest else _fill(_fit(items, prefs["minutes"]), prefs["minutes"], pool, tags)})
    if dropped:
        log.info("workout_plan_dropped_ids", count=dropped)
    tips = [_clean(t, 160) for t in (raw.get("tips") or [])[:3] if t]
    return out, tips


def build_plan(profile: dict, watch: dict | None, prefs_in: dict) -> dict:
    goal_map = {"Lose weight": "weight_loss", "Build muscle": "strength", "Stay fit": "general", "Manage a condition": "low_impact"}
    prefs = {
        "goal": prefs_in.get("goal") if prefs_in.get("goal") in GOALS else goal_map.get(profile.get("goal", ""), "general"),
        "days": int(min(max(_num(prefs_in.get("days"), 4), 2), 7)),
        "minutes": int(min(max(_num(prefs_in.get("minutes"), 25), 10), 90)),
        "level": prefs_in.get("level") if prefs_in.get("level") in LEVELS else "beginner",
        "equipment": [e for e in (prefs_in.get("equipment") or []) if e in ("dumbbells", "chair", "mat", "wall")],
    }
    tags, notes, blocked = health_tags(profile, watch)
    base = {"prefs": prefs, "tags": tags, "tag_labels": [CATALOG["conditionLabels"].get(t, t) for t in tags], "notes": notes, "disclaimer": NOT_ADVICE}
    if blocked:
        return {**base, "blocked": blocked, "days": [], "excluded": [], "tips": [], "source": "rules"}
    if "senior" in tags or "obesity_severe" in tags or "heart" in tags:
        if prefs["goal"] == "weight_loss":
            notes.append("High-impact cardio is replaced with low-impact moves for your safety.")
    pool = {i: e for i, e in allowed(tags, prefs["equipment"], prefs["level"]).items() if e["category"] in GOAL_CATEGORIES[prefs["goal"]]}
    excluded = [{"id": e["id"], "name": e["name"], "reason": ", ".join(CATALOG["conditionLabels"].get(t, t) for t in e["avoid"] if t in tags)}
                for e in EX.values() if set(e["avoid"]) & set(tags)]
    person = {k: profile.get(k) for k in ("age", "sex", "heightCm", "weightKg", "goal", "conditions")}
    days, tips = ai_plan(person, prefs, pool, tags)
    source = "ai"
    rules = rule_plan(prefs["goal"], prefs["days"], prefs["minutes"], pool, tags)
    if not days or not any(not d["rest"] for d in days):
        days, source, tips = rules, "rules", []
    else:
        train = [i for i, d in enumerate(days) if not d["rest"]]
        for i in train[prefs["days"]:]:  # too many training days: keep the first N
            days[i] = rules[i] if rules[i]["rest"] else {"day": DAYS[i], "rest": True, "title": "Rest / light walk", "focus": "Recovery", "items": [], "why": "Rest days let your body recover."}
        missing = prefs["days"] - sum(1 for d in days if not d["rest"])
        for i in TRAIN_DAYS[prefs["days"]]:  # too few: fill from the rule planner
            if missing <= 0:
                break
            if days[i]["rest"]:
                days[i] = {**rules[i], "rest": False} if not rules[i]["rest"] else {"day": DAYS[i], "rest": False, **_rule_day(THEMES[prefs["goal"]][0], pool, tags, prefs["minutes"], i)}
                missing -= 1
    for d in days:
        d["minutes"] = round(sum(item_seconds(i) for i in d["items"]) / 60) if d["items"] else 0
    return {**base, "blocked": None, "days": days, "excluded": excluded, "tips": tips, "source": source}


# ---------- coach grounding ----------
def exercise_facts(exercise_id: str | None) -> str | None:
    e = EX.get(exercise_id or "")
    if not e:
        return None
    return f"{e['name']} ({e['category']}). Correct technique: " + " ".join(e["steps"])


def other_exercise_named(text: str, exercise_id: str | None) -> bool:
    """True if the reply names a different catalog exercise (a sign the model drifted)."""
    low = text.lower()
    me = EX.get(exercise_id or "", {}).get("name", "").lower()
    for e in EX.values():
        n = e["name"].lower().split(" (")[0]
        if len(n) > 6 and n in low and n not in me:
            return True
    return False
