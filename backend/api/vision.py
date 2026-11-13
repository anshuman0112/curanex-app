"""Upload anything: work out what a file is (lab report, chest X-ray, ECG, MRI, skin photo, ...) and which live model should check it.

The free Groq vision model (VISION_MODEL) only RECOGNISES the file and writes a short description. Medical results still come from
the Verified models; when no model fits (ECG, CT, spine X-ray, ...) the app shows the description labelled as an AI reading, not a diagnosis.
"""
import base64
import io
import json
import re
import time

import httpx
from PIL import Image

from api.core.config import get_settings
from api.core.logging import log

VISION_MODEL = "qwen/qwen3.8-27b"
MAX_SIDE = 768  # small request (free tier counts image tokens); enough to recognise the image type
ATTEMPTS, MAX_WAIT_S = 5, 20

PROMPT = ("You look at one uploaded medical file. Say what it is and describe only what is visible. Do not diagnose with certainty. "
          "kind must be one of: lab report, prescription, discharge summary, chest x-ray, knee x-ray, hand or arm x-ray, other x-ray, ecg, "
          "brain mri, other mri, ct scan, ultrasound, skin photo, eye fundus photo, retinal oct, teeth photo, endoscopy, blood smear, other. "
          'Reply only JSON: {"kind": "...", "body_part": "...", "has_text": true/false, "findings": ["short plain-English observation"]}')

# kind -> live model that checks it (first match wins). Kinds not here get the AI description only.
ROUTES = [
    ("chest x-ray", "mlmed_torchxrayvision"),
    ("knee x-ray", "ghifar_khder_knee_osteoarthritis_classification"),
    ("hand or arm x-ray", "joystonmenezes_xray_vision"),
    ("brain mri", "brain_mri_check"),
    ("skin photo", "skin_lesion_check"),
    ("teeth photo", "andreygermanov_yolov8_caries_detector"),
    ("eye fundus photo", "diabetic_retinopathy_check"),
    ("retinal oct", "amd_oct_check"),
    ("ultrasound", "breast_ultrasound_check"),
    ("endoscopy", "endoscopy_check"),
    ("blood smear", "malaria_cell_check"),
]
DOCUMENTS = ("lab report", "prescription", "discharge summary")
BREAST = re.compile(r"breast", re.I)


def _small_jpeg(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def identify(data: bytes) -> dict:
    """{"kind", "body_part", "has_text", "findings"}; kind "unknown" if the vision model is unavailable."""
    s = get_settings()
    base, key = s.llm_endpoint
    url = "data:image/jpeg;base64," + base64.b64encode(_small_jpeg(data)).decode()
    body = {"model": VISION_MODEL, "temperature": 0, "max_tokens": 700,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}]}
    try:
        for attempt in range(ATTEMPTS):  # free tier: wait as long as Groq asks on 429
            r = httpx.post(f"{base.rstrip('/')}/chat/completions", timeout=60, headers={"Authorization": f"Bearer {key}"}, json=body)
            if r.status_code != 429 or attempt == ATTEMPTS - 1:
                break
            wait = min(float(r.headers.get("retry-after") or 2 ** attempt), MAX_WAIT_S)
            log.info("vision_rate_limited", wait_s=wait)
            time.sleep(wait)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        found = re.findall(r"\{.*\}", text, re.S)
        out = json.loads(found[-1]) if found else {}
    except Exception as e:  # recognition is a convenience: never break the upload
        log.warning("vision_identify_error", error=type(e).__name__, detail=str(e)[:200])
        out = {}
    kind = str(out.get("kind") or "unknown").strip().lower()
    findings = [str(f).strip() for f in (out.get("findings") or []) if str(f).strip()][:6]
    return {"kind": kind, "body_part": str(out.get("body_part") or "").strip(), "has_text": bool(out.get("has_text")), "findings": findings}


def route(kind: str, body_part: str = "") -> str | None:
    if kind == "ultrasound" and not BREAST.search(body_part):
        return None  # the only ultrasound model is for breast lumps
    for name, model in ROUTES:
        if kind == name:
            return model
    return None


def is_document(kind: str) -> bool:
    return kind in DOCUMENTS
