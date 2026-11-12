"""Doctor module API (doctor-only). Until accounts and roles are built, every route needs the server's DOCTOR_DEV_KEY in the
X-Doctor-Key header; with no key configured the module is switched off (403). Patients never get this key.
Later the dependency becomes require_role("doctor") from api/auth (see workflow/DOCTOR_MODULE_PLAN_APPROVED.md).
"""

import hmac
import os

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from api.knowledge import answer
from api.knowledge.drugs import medicine_card

router = APIRouter(prefix="/doctor", tags=["doctor"])


def doctor_access(x_doctor_key: str | None = Header(default=None)) -> None:
    key = os.environ.get("DOCTOR_DEV_KEY", "")
    if not key or not x_doctor_key or not hmac.compare_digest(key, x_doctor_key):
        raise HTTPException(status_code=403, detail="Doctor access only.")


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


@router.post("/tests", dependencies=[Depends(doctor_access)])
def tests(case: Case) -> dict:
    return answer.suggest_tests(_case(case))


@router.post("/differential", dependencies=[Depends(doctor_access)])
def differential(case: Case) -> dict:
    return answer.differential(_case(case))


@router.post("/treatment", dependencies=[Depends(doctor_access)])
def treatment(req: TreatmentRequest) -> dict:
    return answer.treatment(_case(req.case), req.disease)


@router.post("/medicine", dependencies=[Depends(doctor_access)])
def medicine(name: str, patient: Patient) -> dict:
    return medicine_card(name, {k: v for k, v in patient.model_dump().items() if v is not None})
