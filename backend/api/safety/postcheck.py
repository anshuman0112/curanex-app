"""Output post-check applied to every AI reply: strip doses, soften "you have X", ensure a "see a doctor" line."""
import re

DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|µg|iu|units?)\b[^.\n]*", re.I)
# "ml" is also used for water and food, so strip it only in sentences about medicines
ML_DOSE = re.compile(r"[^.\n]*\b(syrup|tablet|capsule|medicine|medication|drug|dose|injection|drops?|tonic)\b[^.\n]*", re.I)
ML = re.compile(r"\b\d+(\.\d+)?\s?ml\b[^.\n]*", re.I)
DIAGNOSIS = re.compile(
    r"\byou (?:definitely |certainly |clearly |probably )?(?:have|are suffering from|are diagnosed with)\s+"
    r"(?=(?:an? |the )?[\w\s'-]{0,40}?\b(?:disease|cancer|diabetes|infection|disorder|syndrome|pneumonia|tuberculosis|"
    r"tb|covid|hypertension|anaemia|anemia|tumou?r|failure|condition|psoriasis|caries|arthritis|thyroid)\b)",
    re.I,
)


MARKDOWN = re.compile(r"\*\*|__|^#{1,6}\s*", re.M)  # the app shows plain text


def post_check(text: str) -> str:
    text = MARKDOWN.sub("", text)
    text = DOSE.sub("[dose removed: ask your doctor or pharmacist]", text)
    text = ML_DOSE.sub(lambda m: ML.sub("[dose removed: ask your doctor or pharmacist]", m.group(0)), text)
    text = DIAGNOSIS.sub("you may have signs of ", text)
    if "doctor" not in text.lower():
        text += "\n\nIf symptoms get worse or don't improve, please see a doctor."
    return text
