"""Doctor module API (doctor-only). Access: a logged-in doctor whose account the admin approved (Bearer token), or the server's
DOCTOR_DEV_KEY in X-Doctor-Key (for testing; empty = off). Patients, pending or suspended doctors get 403.
Every answer is rate-limited per user and written to the audit log.
"""

import hmac
import os
import time
from collections import deque
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from api.auth import store
from api.knowledge import answer
from api.knowledge.drugs import medicine_card
from api.knowledge.guard import allow

router = APIRouter(prefix="/doctor", tags=["doctor"])


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _env(name: str) -> str:
    """Value from the environment, else from backend/.env (git-ignored). Tests never read the .env file."""
    if os.environ.get(name):
        return os.environ[name]
    if "PYTEST_CURRENT_TEST" in os.environ or not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


def open_access() -> bool:
    """DOCTOR_OPEN_ACCESS=1: the AI workspace (tests, diseases, treatment, chat) opens without a doctor login (owner, 2026-11-12:
    "doctor panel should be directly accessed for now"). Patient records shared with a doctor still need that doctor's login."""
    return _env("DOCTOR_OPEN_ACCESS") == "1"


def _configured_key() -> str:
    """DOCTOR_DEV_KEY from the environment, else from backend/.env (git-ignored). Empty = module switched off."""
    if os.environ.get("DOCTOR_DEV_KEY"):
        return os.environ["DOCTOR_DEV_KEY"]
    if "PYTEST_CURRENT_TEST" in os.environ or not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("DOCTOR_DEV_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


_fails: deque = deque()
FAIL_LIMIT, FAIL_WINDOW_S = 10, 600  # a 4-digit PIN must not be guessable: 10 wrong tries locks PIN entry for 10 minutes


def _locked() -> bool:
    now = time.time()
    while _fails and now - _fails[0] > FAIL_WINDOW_S:
        _fails.popleft()
    return len(_fails) >= FAIL_LIMIT


def doctor_access(x_doctor_key: str | None = Header(default=None), authorization: str | None = Header(default=None)) -> dict:
    who = None
    if authorization and authorization.lower().startswith("bearer "):
        user = store.user_for_token(authorization.split(" ", 1)[1].strip())
        if user and user["role"] == "doctor" and user["status"] == "approved":
            who = {"id": user["id"], "label": f"user:{user['id']}"}
    if who is None:
        key = _configured_key()
        if x_doctor_key and _locked():
            raise HTTPException(status_code=429, detail="Too many wrong PINs. Try again in 10 minutes.")
        if key and x_doctor_key and hmac.compare_digest(key, x_doctor_key):
            who = {"id": None, "label": "pin"}
        elif x_doctor_key:
            _fails.append(time.time())
    if who is None and open_access():
        who = {"id": None, "label": "open"}
    if who is None:
        raise HTTPException(status_code=403, detail="Doctor access only.")
    if not allow(who["label"]):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a few minutes.")
    return who


def _audit(who: dict, action: str) -> None:
    con = store.connect()
    store.audit(con, who["id"], action)
    con.commit()
    con.close()


class Patient(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = None
    pregnant: bool | None = None
    weight_kg: float | None = Field(default=None, gt=0, le=400)
    kidney_disease: bool | None = None
    liver_disease: bool | None = None
    allergies: list[str] | None = None
    current_medicines: list[str] | None = None


class Case(BaseModel):
    patient: Patient = Patient()
    symptoms: str = Field(min_length=3, max_length=2000)
    history: str | None = Field(default=None, max_length=2000)
    report: dict[str, float | str] | None = None
    model_results: list[str] | None = None


class TreatmentRequest(BaseModel):
    case: Case
    disease: str = Field(min_length=2, max_length=200)


class Turn(BaseModel):
    role: str = Field(max_length=10)
    text: str = Field(max_length=4000)


class AskRequest(BaseModel):
    case: Case
    question: str = Field(min_length=2, max_length=1000)
    history: list[Turn] = Field(default=[], max_length=20)


def _case(c: Case) -> dict:
    d = c.model_dump()
    d["patient"] = {k: v for k, v in d["patient"].items() if v is not None}
    return d


@router.get("/access")
def access() -> dict:
    return {"open": open_access()}


@router.get("/check")
def check(who: dict = Depends(doctor_access)) -> dict:
    """The doctor portal's PIN screen calls this; 403 = wrong PIN."""
    return {"ok": True}


@router.post("/tests")
def tests(case: Case, who: dict = Depends(doctor_access)) -> dict:
    _audit(who, "doctor:ask:tests")
    return answer.suggest_tests(_case(case))


@router.post("/differential")
def differential(case: Case, who: dict = Depends(doctor_access)) -> dict:
    _audit(who, "doctor:ask:differential")
    return answer.differential(_case(case))


@router.post("/treatment")
def treatment(req: TreatmentRequest, who: dict = Depends(doctor_access)) -> dict:
    _audit(who, "doctor:ask:treatment")
    return answer.treatment(_case(req.case), req.disease)


@router.post("/medicine")
def medicine(name: str, patient: Patient, who: dict = Depends(doctor_access)) -> dict:
    _audit(who, "doctor:ask:medicine")
    return medicine_card(name, {k: v for k, v in patient.model_dump().items() if v is not None})


@router.post("/ask")
def ask(req: AskRequest, who: dict = Depends(doctor_access)) -> dict:
    _audit(who, "doctor:ask:chat")
    return answer.ask(_case(req.case), req.question, [t.model_dump() for t in req.history])
