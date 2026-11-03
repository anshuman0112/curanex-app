"""Output post-check applied to every AI reply: strip doses, ensure a "see a doctor" line."""
import re

DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|units?)\b[^.\n]*", re.I)


def post_check(text: str) -> str:
    text = DOSE.sub("[dose removed: ask your doctor or pharmacist]", text)
    if "doctor" not in text.lower():
        text += "\n\nIf symptoms get worse or don't improve, please see a doctor."
    return text
