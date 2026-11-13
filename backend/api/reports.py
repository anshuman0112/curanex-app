"""Reads medical reports (PDF or photo) on this server, for free. Nothing is stored or logged: the text goes back to the app."""
import io
import re
import threading

import numpy as np
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from api.core.config import get_settings

MAX_PAGES = 10
MAX_CHARS = 6000
MIN_CHARS = 20
_ocr = None
_ocr_lock = threading.Lock()

# Report label to model input. Short acronyms stay case-sensitive so "alt" inside a word can't match.
_VALUES = {
    "glucose": r"(?i:fasting (?:blood |plasma )?(?:glucose|sugar)|blood (?:glucose|sugar)|plasma glucose|glucose|sugar)|FBS|FBG|RBS",
    "alt": r"SGPT|\bALT\b|(?i:alanine (?:amino)?transaminase)",
    "ast": r"SGOT|\bAST\b|(?i:aspartate (?:amino)?transaminase)",
    "bilirubin": r"(?i:bilirubin)(?i:[, ]+total)?|(?i:total bilirubin)",
    "bmi": r"\bBMI\b",
    "tsh": r"\bTSH\b|(?i:thyroid stimulating hormone)",
    "triglycerides": r"(?i:triglycerides?)|\bTG\b",
    "hdl": r"\bHDL\b(?i:[- ]?(?:c|cholesterol))?|(?i:hdl[- ]?cholesterol)",
    "bun": r"\bBUN\b|(?i:blood urea nitrogen|urea nitrogen)",
    "uric_acid": r"(?i:uric acid|serum urate)",
}
_NUMBER = r"[^\d\n]{0,40}\n?[^\d\n]{0,20}?(\d{1,4}(?:\.\d+)?)\s*(mmol)?"
# Which check each set of found values can pre-fill.
REPORT_CHECKS = [
    ("nhanes_fatty_liver", {"alt", "ast"}),  # Verified liver check
    ("nhanes_diabetes_any_labs", {"triglycerides", "hdl"}),  # Verified: diabetes from a routine blood test, no sugar input
    ("nhanes_low_egfr_labs", {"bun", "uric_acid"}),  # Verified: reduced kidney function without creatinine
]  # the TSH thyroid check was retired 2026-11-14 (failed its outside test), so TSH no longer pre-fills a check


class ReportRejected(Exception):
    pass


def _ocr_image(img: Image.Image) -> str:
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
        result, _ = _ocr(np.array(img.convert("RGB"))[:, :, ::-1])
    return _rows(result or [])


def _rows(boxes: list) -> str:
    """OCR gives one box per word group. Rebuild table rows, so a test name and its value end up on one line."""
    items = sorted(((sum(p[1] for p in box) / 4, box[0][0], box[2][1] - box[0][1], text) for box, text, _ in boxes), key=lambda i: i[0])
    rows: list[list] = []
    for item in items:
        if rows and abs(item[0] - rows[-1][0][0]) < 0.6 * item[2]:
            rows[-1].append(item)
        else:
            rows.append([item])
    return "\n".join(" ".join(i[3] for i in sorted(row, key=lambda i: i[1])) for row in rows)


def _read_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ReportRejected("This PDF is password protected. Please upload an unlocked copy.")
        pages = reader.pages[:MAX_PAGES]
        text = "\n".join(page.extract_text() or "" for page in pages)
        if len(text.strip()) >= MIN_CHARS:
            return text
        scans = [Image.open(io.BytesIO(im.data)) for page in pages for im in page.images]  # a scanned PDF is pictures of pages
        return "\n".join(_ocr_image(im) for im in scans[:MAX_PAGES])
    except (PdfReadError, UnidentifiedImageError, OSError, ValueError) as e:
        raise ReportRejected("This PDF could not be read.") from e


def read_report(data: bytes) -> str:
    limit = get_settings().upload_max_bytes
    if len(data) > limit:
        raise ReportRejected(f"File is larger than {limit // (1024 * 1024)} MB")
    if data[:5] == b"%PDF-":
        text = _read_pdf(data)
    else:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise ReportRejected("Please upload a PDF or a JPEG, PNG or WebP photo.") from e
        text = _ocr_image(img)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < MIN_CHARS:
        raise ReportRejected("I couldn't find any written text. If this is an X-ray or scan, close this and tap the matching check under 'Upload an image' (for example Chest X-ray). For a lab report, upload a clearer, straight-on photo.")
    return text[:MAX_CHARS]


def find_values(text: str) -> dict[str, float]:
    """Lab values for pre-filling a form. The user always sees and can correct them before a check runs."""
    found = {}
    for key, label in _VALUES.items():
        m = re.search(f"(?:{label}){_NUMBER}", text)
        if m and not m.group(2):  # the models expect mg/dL, not mmol/L
            found[key] = float(m.group(1))
    return found
