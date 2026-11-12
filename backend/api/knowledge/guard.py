"""Guardrails for the doctor module.

Input:  length limits, control characters removed, prompt-injection phrases neutralised, identity details (names, phone numbers,
        emails, Aadhaar/ID-like numbers) removed before anything is sent to the AI; case text is fenced as data, not instructions.
Output: no dose may appear in AI-written plan or referral text (doses come only from official labels via drugs.py);
        every statement is checked against its cited passage (verify.py).
Abuse:  per-user request limit.
"""

import re
import time
from collections import defaultdict, deque

MAX_FIELD = 2000
INJECTION = re.compile(r"(ignore (all |any )?(previous|prior|above) (instructions|rules)|disregard (the )?(rules|instructions)|"
                       r"you are now|act as|system prompt|developer message|new instructions|jailbreak|</?(system|case|passages)>)", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
ID_NUMBER = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")  # Aadhaar-like 12 digits
NAME_INTRO = re.compile(r"\b(my name is|name is|patient name|named|mr\.?|mrs\.?|ms\.?|shri|smt\.?)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?", re.I)
DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|µg|g|ml|mL|units?|iu|mg/kg)\b|\b(once|twice|thrice|\d+ times) (a|per) day\b|\bevery \d+\s*(h|hours)\b", re.I)
WINDOW_S, MAX_CALLS = 600, 40
_calls: dict[str, deque] = defaultdict(deque)


def clean(text: str | None, limit: int = MAX_FIELD) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text or ""))[:limit]
    text = INJECTION.sub("[removed]", text)
    return re.sub(r"\s+", " ", text).strip()


def deidentify(text: str) -> str:
    text = EMAIL.sub("[email]", text)
    text = ID_NUMBER.sub("[id]", text)
    text = PHONE.sub("[phone]", text)
    return NAME_INTRO.sub(lambda m: m.group(1) + " [name]", text)


def safe_case(case: dict) -> dict:
    """The case as the AI may see it: cleaned, de-identified, size-limited."""
    out = dict(case)
    for k in ("symptoms", "history"):
        if out.get(k):
            out[k] = deidentify(clean(out[k]))
    if out.get("report"):
        out["report"] = {clean(k, 60): deidentify(clean(v, 200)) for k, v in list(out["report"].items())[:40]}
    if out.get("model_results"):
        out["model_results"] = [deidentify(clean(m, 300)) for m in out["model_results"][:10]]
    return out


def no_dose(items: list[dict]) -> tuple[list[dict], int]:
    """Drop AI-written statements that contain a dose; doses are shown only from official labels."""
    kept = [i for i in items if not DOSE.search(str(i.get("text", "")))]
    return kept, len(items) - len(kept)


def allow(user: str) -> bool:
    """Sliding-window limit per user/key: MAX_CALLS answers per WINDOW_S seconds."""
    now, q = time.time(), _calls[user]
    while q and now - q[0] > WINDOW_S:
        q.popleft()
    if len(q) >= MAX_CALLS:
        return False
    q.append(now)
    return True
