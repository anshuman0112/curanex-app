"""Symptom engine: possible causes, urgency, tests and self-care for any symptom, whether or not a trained model exists.

The AI drafts the plan, then plain code decides what the patient may see. Red flags have already been handled before this
runs. Medicines can only come from the OTC list below, never with a dose, and the profile can remove any of them.
"""
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from api.clients.llm import chat_completion
from api.core.logging import log
from api.orchestrator.prompts import CARE
from api.safety.postcheck import DOSE, scrub
from api.schemas.chat import ChatRequest

MAY_ASK = ("Like a doctor taking a history: when this is the first message about a symptom and important details are missing, ask 3-5 short questions, each with 2-4 tap-to-answer options (for fever: how many days, how high, chills, dizziness or weakness, body ache, cough or cold, vomiting or loose motions, rash, pain while passing urine). Options must be short, e.g. [\"Yes\", \"No\"] or [\"1-2 days\", \"3-5 days\", \"More than 5 days\"]. If you ask, leave the other keys empty. If the patient already gave enough detail, leave this empty and give the full plan.")
NO_ASK = "Leave this empty. Do not ask anything. Give the full plan now, filling in every other key, using what the patient has said."
ASKED = "To guide you better"  # opens the follow-up questions, so the next reply continues this flow
CHILD_AGE = 12
CARE_YOUNG = 5
CARE_OLD = 65
RANK = {"self_care": 0, "doctor_soon": 1, "doctor_today": 2, "emergency": 3}
URGENCY = {
    "self_care": "Home care is usually enough",
    "doctor_soon": "See a doctor in the next few days",
    "doctor_today": "See a doctor today",
    "emergency": "Go to emergency care now",
}
SYMPTOM = re.compile(
    r"\b(pain|ache|aching|aches|hurts?|hurting|sore|fever|cough\w*|cold|flu|vomit\w*|nausea|dizz\w*|rash|itch\w*|swell\w*|swollen|"
    r"diarrh\w*|loose motion\w*|constipat\w*|tired|fatigue|weak\w*|headache|migraine|bleed\w*|burning|cramps?|thirsty|urinat\w*|"
    r"numb\w*|breath\w*|wheez\w*|insomnia|can'?t sleep|anxious|bloat\w*|acidity|heartburn|gas|sneez\w*|runny nose|symptoms?|painful|"
    r"yellow|jaundice|dark urine|appetite|lump|stiff\w*|faint\w*|palpitation\w*|sweat\w*|pale|swallow\w*|patch\w*|"
    r"i feel|i am feeling|i'?ve been|i have been|i have had|my \w+ (?:is|are|feels?|keeps?)|"
    r"bukhar|dard|khansi|ulti|chakkar|jukam|sardi|kamzori)\b|बुखार|दर्द|खांसी|खाँसी|उल्टी|चक्कर|ज़ुकाम|जुकाम|कमज़ोरी|कमजोरी|सूजन|खुजली|दस्त", re.I)
# Prescription drugs by name or family ending. Only allowed in a plan if the patient already takes that drug.
RX = re.compile(r"\b(\w+cillin|\w+mycin|\w+floxacin|\w+prazole|\w+sartan|\w+olol|\w+dipine|\w+pril|\w+statin|cef\w+|prednis\w+|dexamethasone|"
                r"metformin|glimepiride|tramadol|diazepam|alprazolam|ondansetron|domperidone|doxycycline|metronidazole|hydroxychloroquine|levothyroxine|warfarin)\b", re.I)

# Causes that need a doctor the same day, whatever urgency the AI chose. Not lowered by the AI.
SERIOUS = re.compile(r"dengue|malaria|typhoid|tubercul|pneumonia|appendic|sepsis|thrombosis|dvt|clot|pre-?eclampsia|hepatitis|jaundice|meningitis|"
                     r"stroke|stone|renal colic|ectopic|angina|myocardial|heart attack|cholecyst|pancreat|encephal|delirium", re.I)
SURGICAL = re.compile(r"appendic|stone|renal colic|thrombosis|dvt|clot|ectopic|cholecyst|pancreat|ulcer|bleed", re.I)
LONG = re.compile(r"\b(\d+|a|two|three|four|five|six|several|few|many)\s+(weeks?|months?|years?)\b|every (week|month)", re.I)
LONG_FEVER = re.compile(r"\b([3-9]|\d\d)\s*[-–]\s*\d+\s*(din|days?)\b|(?<![-–])\b([3-9]|\d\d|three|four|five|six|seven|teen|chaar|paanch)\s*(din|days?)\b|\b(a|one|\d+)\s+weeks?\b", re.I)
FEVER = re.compile(r"fever|bukhar|chills|बुखार", re.I)
# "no fever", "koi bukhar nahi", "बुखार नहीं": removed before looking for fever, so a denied fever doesn't raise urgency
NO_FEVER = re.compile(r"\b(?:no|without|koi|never had)\s+(?:any\s+|high\s+)?(?:fever|bukhar|chills)(?:\s+(?:nahi|nahin|nhi|na))?\b|"
                      r"\b(?:fever|bukhar)\s+(?:nahi|nahin|nhi)\b|\bnahi\s+(?:hai\s+)?bukhar\b|बुखार\s*नहीं|कोई\s*बुखार\s*नहीं", re.I)


def has_fever(text: str) -> bool:
    return bool(FEVER.search(NO_FEVER.sub(" ", text)))
# Medicine: words in the symptoms or causes that make it relevant
USE_FOR = {
    "Paracetamol": r"fever|bukhar|pain|ache|hurts?|dard|sore|headache|migraine",
    "Ibuprofen": r"pain|ache|hurts?|dard|sprain|strain|migraine",
    "Oral rehydration salts (ORS)": r"diarrh|loose motion|vomit|ulti|dehydrat|gastroenter",
    "Cetirizine": r"allerg|sneez|itch|rash|hives|runny nose",
    "Antacid": r"acid|heartburn|reflux|gerd|burning in my chest|indigestion|gastritis",
    "Saline nasal drops": r"nose|nasal|sinus|congest|cold",
    "Throat lozenges": r"throat",
    "Calamine lotion": r"itch|rash|sunburn|chickenpox",
    "Clotrimazole cream": r"fung|tinea|ringworm|jock itch",
    "Psyllium husk (isabgol)": r"constipat|hard stool",
    "Loratadine": r"allerg|sneez|itch|hives|runny nose|hay fever",
    "Dextromethorphan cough syrup": r"dry cough|cough",
    "Ambroxol cough syrup": r"wet cough|phlegm|sputum|mucus|chest congestion|cough",
    "Honey with warm water": r"cough|throat|khansi",
    "Steam inhalation": r"cold|nose|nasal|sinus|congest|cough|jukam|sardi",
    "Menthol vapour rub": r"cold|nose|congest|blocked nose|headache|body ache|jukam",
    "Diclofenac gel": r"sprain|strain|muscle|back pain|neck pain|joint pain|knee pain|shoulder pain|body ache",
    "Simethicone": r"gas|bloat|flatulence|wind",
    "Oral zinc with ORS": r"diarrh|loose motion",
}
EARLY_NOTE = "Please get checked first. The doctor will tell you which medicine is safe for you."

# Name: (other names, words in the profile or message that remove it, minimum age)
OTC: dict[str, tuple[tuple[str, ...], tuple[str, ...], int]] = {
    "Paracetamol": (("acetaminophen", "crocin", "dolo"), ("liver", "alcohol"), 12),
    "Ibuprofen": (("brufen",), ("pregnan", "kidney", "ulcer", "asthma", "heart", "hypertension", "blood pressure", "warfarin", "aspirin", "blood thinner", "fever", "bukhar", "dengue", "stomach", "acidity", "heartburn", "gastr", "vomit"), 12),
    "Oral rehydration salts (ORS)": (("ors", "oral rehydration"), ("kidney",), 1),
    "Cetirizine": (("cetzine",), ("pregnan", "kidney"), 12),
    "Antacid": (("antacid", "gelusil", "digene"), ("kidney",), 12),
    "Saline nasal drops": (("saline", "nasal saline"), (), 1),
    "Throat lozenges": (("lozenge",), (), 12),
    "Calamine lotion": (("calamine",), (), 1),
    "Clotrimazole cream": (("clotrimazole",), ("pregnan",), 12),
    "Psyllium husk (isabgol)": (("psyllium", "isabgol"), ("swallowing",), 12),
    "Loratadine": (("loratadine", "lorfast"), ("pregnan", "liver"), 12),
    "Dextromethorphan cough syrup": (("dextromethorphan", "dry cough syrup"), ("pregnan", "asthma", "antidepress", "ssri", "maoi", "breathless", "wheez"), 12),
    "Ambroxol cough syrup": (("ambroxol", "mucolite"), ("pregnan", "ulcer"), 12),
    "Honey with warm water": (("honey",), ("diabet",), 1),
    "Steam inhalation": (("steam",), (), 6),
    "Menthol vapour rub": (("vapour rub", "vaporub", "vicks", "menthol"), ("asthma",), 2),
    "Diclofenac gel": (("diclofenac", "volini", "moov"), ("pregnan", "aspirin allerg", "nsaid allerg"), 12),
    "Simethicone": (("simethicone",), (), 12),
    "Oral zinc with ORS": (("zinc",), (), 1),
}


def is_care_turn(req: ChatRequest) -> bool:
    if req.history and req.history[-1].role == "assistant" and req.history[-1].content.startswith(ASKED):
        return True
    return bool(SYMPTOM.search(req.message or ""))


def _facts(req: ChatRequest) -> tuple[str, float | None]:
    p = (req.context or {}).get("profile") or {}
    words = [str(x) for k in ("conditions", "allergies", "medications") for x in (p.get(k) or [])]
    if p.get("pregnant"):
        words.append("pregnant")
    try:
        age = float(p["age"])
    except (KeyError, TypeError, ValueError):
        age = None
    return " ".join([*words, req.message or "", *(t.content for t in req.history if t.role == "user")]).lower(), age


def set_urgency(said: str, causes: list[dict], req: ChatRequest) -> str:
    """The AI's urgency can only be raised here, never lowered."""
    urgency = said if said in URGENCY else "doctor_soon"
    facts, age = _facts(req)
    likely = " ".join(c["name"] for c in causes if "less" not in c.get("likelihood", "").lower())
    fever = has_fever(facts)
    floor = "self_care"
    if LONG.search(facts):
        floor = "doctor_soon"
    if SERIOUS.search(likely) or "pregnan" in facts or (fever and LONG_FEVER.search(facts)) or (fever and age is not None and (age < CARE_YOUNG or age >= CARE_OLD)):
        floor = "doctor_today"
    return floor if RANK[floor] > RANK[urgency] else urgency


def safe_medicines(items: list, req: ChatRequest, causes: str = "", urgency: str = "doctor_soon") -> tuple[list[dict], str | None]:
    """Keeps only listed OTC medicines this patient can use. Returns them plus a note when the answer was changed."""
    facts, age = _facts(req)
    facts = f"{facts} {causes.lower()}"
    if urgency == "emergency":
        return [], None
    if age is not None and age < CHILD_AGE:
        return [], "Medicines for children depend on age and weight, so please ask a paediatrician first."
    if "pregnan" in facts:
        return [], "Please check with your doctor before taking any medicine during pregnancy."
    today = urgency == "doctor_today"
    if today and (SURGICAL.search(causes) or not has_fever(facts)):
        return [], EARLY_NOTE
    kept, removed = [], False
    for item in items[:5] if isinstance(items, list) else []:
        said = f"{(item or {}).get('name', '')}".lower() if isinstance(item, dict) else ""
        match = next((name for name, (other, _, _) in OTC.items() if name.lower() in said or any(o in said for o in other)), None)
        if not match:
            continue
        _, avoid, _ = OTC[match]
        if today and match != "Paracetamol":  # while waiting to be seen, only a fever medicine that doesn't hide bleeding
            continue
        if not re.search(USE_FOR[match], facts, re.I):
            continue
        if any(w in facts for w in avoid) or said in facts or match.split(" (")[0].lower() in facts:
            removed = True
            continue
        text = " ".join(str(item.get(k, "")) for k in ("use", "caution"))
        if DOSE.search(text):
            continue
        kept.append({"name": match, "use": scrub(str(item.get("use", "")))[:160], "caution": scrub(str(item.get("caution", "")))[:160]})
    note = "Some options were left out because of your health profile. Please ask your doctor." if removed else None
    return kept[:3], note


def names_rx(text: str, req: ChatRequest) -> bool:
    """True when the text names a prescription drug the patient doesn't already take."""
    facts, _ = _facts(req)
    return any(m.group(1).lower() not in facts for m in RX.finditer(text))


def _reads(items, keys: tuple[str, ...], limit: int, req: ChatRequest) -> list:
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not item.get(keys[0]) or names_rx(" ".join(str(v) for v in item.values()), req):
            continue
        out.append(scrub(str(item[keys[0]]))[:200] if len(keys) == 1 else {k: scrub(str(item.get(k, "")))[:200] for k in keys})
    return out[:limit]


def _questions(items, req: ChatRequest) -> list[dict]:
    """Doctor-style follow-up questions, each with tap-to-answer options (max 5 questions, 4 options)."""
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not item.get("q") or names_rx(str(item), req):
            continue
        options = [scrub(str(o))[:40] for o in (item.get("options") or []) if str(o).strip()][:4] or ["Yes", "No"]
        out.append({"q": scrub(str(item["q"]))[:160], "options": options})
    return out[:5]


def _source(cause: str) -> dict | None:
    """A trusted page for the patient to read. Only the condition name leaves the server, never patient details."""
    try:
        r = httpx.get("https://wsearch.nlm.nih.gov/ws/query", params={"db": "healthTopics", "term": cause, "retmax": 1}, timeout=3)
        url = re.search(r'url="(https://medlineplus\.gov/[^"]+)"', r.text)
        title = re.search(r'<content name="title">(.*?)</content>', r.text, re.S)
        if url and title:
            return {"title": re.sub(r"<[^>]+>", "", html.unescape(title.group(1))), "url": url.group(1)}
    except httpx.HTTPError:
        pass
    return None


def _sources(causes: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(2) as pool:
        found = [s for s in pool.map(_source, [c["name"] for c in causes[:2]]) if s]
    return list({s["url"]: s for s in found}.values())


def _ask_llm(req: ChatRequest, may_ask: bool) -> dict | None:
    profile = json.dumps(req.context or {}, ensure_ascii=False)[:3000]
    system = CARE.format(medicines=", ".join(OTC), profile=profile, ask=MAY_ASK if may_ask else NO_ASK)
    msgs = [{"role": "system", "content": system}, *[t.model_dump() for t in req.history[-6:]], {"role": "user", "content": req.message}]
    msg = chat_completion(msgs, json_mode=True, max_tokens=1300)
    try:
        return json.loads((msg or {}).get("content") or "")
    except json.JSONDecodeError:
        log.warning("care_plan_unreadable")
        return None


def care_blocks(req: ChatRequest, may_ask: bool = True) -> tuple[list[dict], bool] | None:
    """(blocks, is_full_plan), or None when the AI gave nothing usable, so the normal path can answer instead."""
    followup = bool(req.history and req.history[-1].role == "assistant" and req.history[-1].content.startswith(ASKED))
    plan = _ask_llm(req, may_ask and not followup)
    if not isinstance(plan, dict):
        return None
    questions = _questions(plan.get("questions"), req) if may_ask and not followup else []
    if questions:
        return [{"type": "text", "text": f"{ASKED}, please answer a few quick questions (tap an answer):"},
                {"type": "questions", "questions": questions}], False
    causes = _reads(plan.get("causes"), ("name", "why", "likelihood"), 4, req)
    if not causes:  # it asked again instead of answering
        plan = _ask_llm(req, False)
        causes = _reads((plan or {}).get("causes"), ("name", "why", "likelihood"), 4, req) if isinstance(plan, dict) else []
    if not causes:
        return None
    urgency = set_urgency(plan.get("urgency"), causes, req)
    medicines, note = safe_medicines(plan.get("medicines"), req, " ".join(c["name"] for c in causes), urgency)
    summary = str(plan.get("summary") or "")
    summary = "" if names_rx(summary, req) else scrub(summary)[:400]
    card = {
        "type": "care_plan", "urgency": urgency, "urgency_label": URGENCY[urgency], "causes": causes,
        "tests": _reads(plan.get("tests"), ("name", "why"), 4, req), "medicines": medicines, "medicine_note": note,
        "self_care": _reads(plan.get("self_care"), ("x",), 4, req), "see_doctor_if": _reads(plan.get("see_doctor_if"), ("x",), 4, req),
        "sources": _sources(causes),
        "basis": "AI guidance from general medical knowledge. It isn't a tested model or a diagnosis, and it can be wrong.",
    }
    text = summary or f"This may be {causes[0]['name']}. {URGENCY[urgency]}."
    return [{"type": "text", "text": text}, card], True
