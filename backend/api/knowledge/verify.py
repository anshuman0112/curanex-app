"""Evidence check for the doctor module: a statement is shown only if the passage it cites supports it.

Rules (strict on purpose):
- the statement must cite a passage that was actually retrieved for this answer;
- every number in the statement (doses, values, days) must appear in that passage, with the same unit if one is given;
- most of the statement's medical words must appear in that passage (word-stem overlap >= MIN_OVERLAP).
Statements that fail are dropped, never repaired. The check is cheap and deterministic, so it is fully testable.
"""

import re

MIN_OVERLAP = 0.6
STOP = set("""a an the and or of in on to for with without by from as at is are was were be been being it its this that these those
which who whom whose what when where why how not no can may might should would could will shall do does did done than then there
their they them he she his her you your our we us i me my patient patients often usually also more most less such other some any
each per into about over under after before during within""".split())
NUMBER = re.compile(r"(\d+(?:\.\d+)?)\s*(mg/kg|mg|mcg|µg|g|ml|mL|units?|iu|%|days?|weeks?|hours?|times?)?", re.I)


def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("ations", "ation", "ments", "ment", "ness", "ities", "ity", "ings", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "κ": "kappa", "µ": "micro", "μ": "micro"}


def normalise(text: str) -> str:
    """Same text whatever hyphen or Greek letter the AI or the book used (CURB‑65 = CURB-65, β-lactam = beta-lactam)."""
    text = re.sub(r"[‐-―−­]", "-", text or "")
    for g, word in GREEK.items():
        text = text.replace(g, word)
    return text


def _words(text: str) -> set[str]:
    return {_stem(p) for w in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", normalise(text)) for p in w.split("-") if len(p) > 2 and p.lower() not in STOP}


def numbers_supported(statement: str, passage: str) -> bool:
    statement, passage = normalise(statement), normalise(passage)
    found = {(n.rstrip("0").rstrip(".") if "." in n else n, (u or "").lower().rstrip("s")) for n, u in NUMBER.findall(passage)}
    plain = {n for n, _ in found}
    for n, u in NUMBER.findall(statement):
        n = n.rstrip("0").rstrip(".") if "." in n else n
        unit = (u or "").lower().rstrip("s")
        if n not in plain or (unit and (n, unit) not in found):
            return False
    return True


def overlap(statement: str, passage: str) -> float:
    s = _words(statement)
    return len(s & _words(passage)) / len(s) if s else 0.0


def supported(statement: str, passage: str) -> bool:
    return bool(statement.strip()) and numbers_supported(statement, passage) and overlap(statement, passage) >= MIN_OVERLAP


def keep_supported(items: list[dict], passages: dict[str, dict], text_key: str = "text") -> tuple[list[dict], int]:
    """items: [{text, cite: [passage ids]}]. Keeps items whose text is supported by at least one cited, retrieved passage.
    Returns (kept items with resolved citations, number dropped)."""
    kept, dropped = [], 0
    for item in items or []:
        if not isinstance(item, dict):
            dropped += 1
            continue
        text = str(item.get(text_key) or "")
        cites = [c for c in (item.get("cite") or []) if c in passages]
        good = [c for c in cites if supported(text, passages[c]["text"])]
        if good:
            kept.append({**item, "cite": good})
        else:
            dropped += 1
    return kept, dropped
