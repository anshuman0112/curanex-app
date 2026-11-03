"""Output post-check applied to every AI reply: strip doses, soften "you have X", ensure a "see a doctor" line."""
import re

DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|units?)\b[^.\n]*", re.I)
DIAGNOSIS = re.compile(
    r"\byou (?:definitely |certainly |clearly |probably )?(?:have|are suffering from|are diagnosed with)\s+"
    r"(?=(?:an? |the )?[\w\s'-]{0,40}?\b(?:disease|cancer|diabetes|infection|disorder|syndrome|pneumonia|tuberculosis|"
    r"tb|covid|hypertension|anaemia|anemia|tumou?r|failure|condition|psoriasis|caries|arthritis|thyroid)\b)",
    re.I,
)


def post_check(text: str) -> str:
    text = DOSE.sub("[dose removed: ask your doctor or pharmacist]", text)
    text = DIAGNOSIS.sub("you may have signs of ", text)
    if "doctor" not in text.lower():
        text += "\n\nIf symptoms get worse or don't improve, please see a doctor."
    return text
