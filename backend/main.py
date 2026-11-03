"""
Curanex backend (FastAPI). Minimal working version of the API contract in ARCHITECTURE.md §6.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

What works now:
    GET  /health
    POST /chat     safety check -> LLM reply as "blocks"  (falls back to a canned reply if no LLM key)
    POST /upload   stores an image, returns image_id
    GET  /models   the model registry

What comes next (ARCHITECTURE.md §2-§5): tool retrieval over model_registry.json,
adapters/ for each ML model, and the orchestrator tool loop. Plug them into `orchestrate()`.
"""
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
REGISTRY = BASE / "model_registry.json"

app = FastAPI(title="Curanex API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- Safety (deterministic, runs before any AI) ----------------
RED_FLAGS: list[tuple[str, str]] = [
    (r"(chest|heart).{0,40}(pain|pressure|tight|crush).{0,60}(arm|jaw|neck|back|sweat|breath)|crushing chest|heart attack", "Possible heart attack"),
    (r"(face|mouth).{0,20}droop|slurred speech|can'?t (move|feel) (my )?(arm|leg|side)|stroke", "Possible stroke"),
    (r"can'?t breathe|cannot breathe|struggling to breathe|choking|lips (are )?(turning )?blue", "Severe breathing difficulty"),
    (r"unconscious|passed out|not responding|unresponsive", "Loss of consciousness"),
    (r"(heavy|severe|won'?t stop|uncontrolled) bleeding|bleeding (a lot|heavily)", "Severe bleeding"),
    (r"seizure|convulsion", "Seizure"),
    (r"(kill|end|hurt) (myself|my life)|suicid|don'?t want to (live|be alive)", "Thoughts of self-harm"),
    (r"overdose|took too many (pills|tablets)|poison(ed|ing)", "Possible overdose or poisoning"),
]


def red_flag(text: str) -> str | None:
    for pattern, reason in RED_FLAGS:
        if re.search(pattern, text, re.I):
            return reason
    return None


DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|units?)\b[^.\n]*", re.I)


def post_check(text: str) -> str:
    text = DOSE.sub("[dose removed: ask your doctor or pharmacist]", text)
    if "doctor" not in text.lower():
        text += "\n\nIf symptoms get worse or don't improve, please see a doctor."
    return text


SYSTEM = """You are Curanex, an AI health guide (not a doctor).
- Never give a definitive diagnosis. Say "possible causes" or "this may be".
- Never give medicine doses or tell someone to start/stop a prescription.
- Use the user's profile (conditions, medicines, allergies, recent meals and workouts) when relevant.
- Be warm, clear and brief. Plain language. Indian context (foods, 112/108 emergency numbers).
- End with when to see a doctor."""

# ---------------- Schemas ----------------
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = ""
    context: dict[str, Any] | None = None
    history: list[Turn] = []
    form_values: dict[str, str] | None = None
    image_id: str | None = None
    model_id: str | None = None


# ---------------- LLM ----------------
def call_llm(messages: list[dict], system: str) -> str | None:
    """Anthropic by default. Set LLM_API_KEY (and optionally LLM_MODEL) in the environment."""
    key = os.getenv("LLM_API_KEY")
    if not key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=700, system=system, messages=messages)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    except Exception as e:  # keep the app usable if the provider fails
        print("LLM error:", e)
        return None


def orchestrate(req: ChatRequest) -> list[dict]:
    """Plug the ML router in here (tool retrieval -> model tools -> explanation)."""
    system = SYSTEM + "\n\nUser context:\n" + json.dumps(req.context or {}, ensure_ascii=False)[:4000]
    msgs = [t.model_dump() for t in req.history[-10:]] + [{"role": "user", "content": req.message or "(user sent inputs)"}]
    text = call_llm(msgs, system)
    if text is None:
        text = ("The AI model isn't configured on the server yet (set LLM_API_KEY). "
                "The app, safety checks and SOS work; answers will appear once the key is added.")
    return [{"type": "text", "text": post_check(text)}]


# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest):
    reason = red_flag(req.message or "")
    if reason:
        return {"blocks": [{"type": "escalate", "reason": reason}]}
    return {"blocks": orchestrate(req)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    image_id = uuid.uuid4().hex
    (UPLOADS / f"{image_id}.jpg").write_bytes(await file.read())
    return {"image_id": image_id}


@app.get("/models")
def models():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"models": []}
