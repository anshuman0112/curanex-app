"""Gym coach voice: /v1/gym/coach turns a workout event (+ form issue) into a short spoken cue.
Port of shradha-khapra/ai-gym-coach services/coaching (LLMCoach + gTTS VoicePipeline), using the
app's free Groq LLM (api/clients/llm.py) instead of the retired llama-3.3. Pose detection runs in the
browser; only the event name and a one-line issue text reach this endpoint (no video, no images)."""
import base64
from io import BytesIO
from typing import Literal

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from typing import Any

from api import workout
from api.clients.llm import chat_completion
from api.core.logging import log

router = APIRouter(tags=["gym"])

PROMPT = (
    "You are Apna AI Coach, a professional AI gym trainer monitoring a user's workout via live camera.\n\n"
    "### Your Role\n"
    "Provide around 10-15 words, high-energy coaching cues. You speak these aloud, so they must be natural and encouraging.\n\n"
    "### Input Format\n"
    "You receive updates in the format: 'Event: [state] Exercise: [name] Form Issue: [description]'.\n"
    "- 'Event': workout_started, set_completed, workout_completed, no_pose_detected, ongoing_form_check.\n"
    "- 'Form Issue': A technical description of a pose error (if any).\n\n"
    "### Guidelines\n"
    "1. Provide feedback in natural, short sentences. Avoid overly brief or fragmented responses.\n"
    "2. NO generic greetings or redundant questions. Focus on the workout.\n"
    "3. Use the second person (e.g., 'Straighten your back' instead of 'The user should straighten their back').\n"
    "4. Maintain a professional coaching tone and prioritize safety. Never give medical advice.\n"
    "5. Reply with the spoken cue only: no quotes, emojis, markdown or stage directions. Only use the technique facts given in <exercise_facts>; never mention a different exercise or numbers you were not given.\n\n"
    "### Scenario Response Styles\n"
    "- 'workout_started' -> A motivating and sharp command to begin.\n"
    "- 'workout_completed' -> A warm and encouraging closing for the session.\n"
    "- 'set_completed' -> Direct praise for finishing the set.\n"
    "- 'no_pose_detected' -> A clear instruction for the user to reposition within the camera frame.\n"
    "- 'ongoing_form_check' + Form Issue -> A precise, supportive correction for the detected error.\n"
    "- 'ongoing_form_check' (No Issue) -> Brief, energetic words of encouragement.\n"
)

# Used when the LLM is not configured or fails, so the coach always speaks.
FALLBACK = {
    "workout_started": "Let's go! Get into position and start your first rep.",
    "set_completed": "Set complete, great work! Take a short breath and keep going.",
    "workout_completed": "Workout complete. Excellent effort today, you should be proud.",
    "no_pose_detected": "I can't see you. Step back so your whole body is in the camera.",
    "ongoing_form_check": "Looking strong, keep that steady pace.",
}

Event = Literal["workout_started", "set_completed", "workout_completed", "no_pose_detected", "ongoing_form_check"]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=400)


class CoachIn(BaseModel):
    event: Event
    exercise: str = Field("", max_length=80)
    exercise_id: str | None = Field(None, max_length=60)
    issue: str | None = Field(None, max_length=300)
    cue: str | None = Field(None, max_length=200)  # plain second-person correction, spoken if the LLM is down
    history: list[Turn] = Field(default_factory=list, max_length=10)
    voice: bool = True


def _tts(text: str) -> str | None:
    """gTTS mp3 as base64 (same engine as the original app). None -> the app falls back to device speech."""
    try:
        from gtts import gTTS
        buf = BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:  # offline, package missing, Google rate limit
        log.warning("gym_tts_failed", error=type(e).__name__)
        return None


def _coach(body: CoachIn) -> dict:
    prompt = f"Event: {body.event}"
    if body.exercise:
        prompt += f" Exercise: {body.exercise}"
    if body.issue:
        prompt += f" Form Issue: {body.issue}"
    facts = workout.exercise_facts(body.exercise_id)
    if facts:
        prompt += f"\n<exercise_facts>{facts}</exercise_facts>"
    messages = [{"role": "system", "content": PROMPT}, *[t.model_dump() for t in body.history[-10:]], {"role": "user", "content": prompt}]
    msg = chat_completion(messages, max_tokens=400)
    text = ((msg or {}).get("content") or "").strip().strip('"')
    text = text.translate({0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: ", ", 0x2019: "'", 0x201C: '"', 0x201D: '"'})
    source = "llm"
    # Guard against drift: too long, or naming a different exercise -> use the deterministic cue instead
    if text and (len(text.split()) > 30 or workout.other_exercise_named(text, body.exercise_id)):
        log.info("gym_coach_reply_rejected", words=len(text.split()))
        text = ""
    if not text:
        text, source = (body.cue if body.issue and body.cue else FALLBACK[body.event]), "fallback"
    return {"text": text, "audio_b64": _tts(text) if body.voice else None, "audio_mime": "audio/mpeg", "source": source}


@router.post("/v1/gym/coach")
async def gym_coach(body: CoachIn) -> dict:
    return await run_in_threadpool(_coach, body)


class PlanIn(BaseModel):
    profile: dict[str, Any] = {}
    watch: dict[str, Any] | None = None   # latest smartwatch day: bp_sys, bp_dia, hr_resting
    prefs: dict[str, Any] = {}            # goal, days, minutes, level, equipment


@router.post("/workout-plan")
@router.post("/v1/workout-plan")
async def workout_plan(body: PlanIn) -> dict:
    """7-day plan from the exercise catalog, filtered for the person's health details (see api/workout.py)."""
    return await run_in_threadpool(workout.build_plan, body.profile, body.watch, body.prefs)


@router.get("/v1/exercises")
def exercises() -> dict:
    return workout.CATALOG
