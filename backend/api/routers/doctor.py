"""Doctor module API (doctor-only). Access: a logged-in doctor whose account the admin approved (Bearer token), or the server's
DOCTOR_DEV_KEY in X-Doctor-Key (for testing; empty = off). Patients, pending or suspended doctors get 403.
Every answer is rate-limited per user and written to the audit log.
"""

import hmac
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from api.auth import store
from api.knowledge import answer
from api.knowledge.drugs import medicine_card
from api.knowledge.guard import allow

router = APIRouter(prefix="/doctor", tags=["doctor"])


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


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


def doctor_access(x_doctor_key: str | None = Header(default=None), authorization: str | None = Header(default=None)) -> dict:
    who = None
    if authorization and authorization.lower().startswith("bearer "):
        user = store.user_for_token(authorization.split(" ", 1)[1].strip())
        if user and user["role"] == "doctor" and user["status"] == "approved":
            who = {"id": user["id"], "label": f"user:{user['id']}"}
    if who is None:
        key = _configured_key()
        if key and x_doctor_key and hmac.compare_digest(key, x_doctor_key):
            who = {"id": None, "label": "dev-key"}
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


def _case(c: Case) -> dict:
    d = c.model_dump()
    d["patient"] = {k: v for k, v in d["patient"].items() if v is not None}
    return d


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
