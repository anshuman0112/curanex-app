"""Health records. Patients keep cases and share them with a doctor (consent, revocable). Approved doctors see only cases shared with
them and confirm a plan; the patient then sees "Your doctor has advised". Contents are encrypted at rest; every access is audited."""

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import store
from api.auth.deps import require_role

router = APIRouter(tags=["records"])


class NewCase(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    symptoms: str = Field(min_length=3, max_length=2000)
    history: str | None = Field(default=None, max_length=2000)
    report: dict[str, str] | None = None
    model_results: list[str] | None = None
    patient: dict | None = None  # age, sex, pregnant, weight_kg, kidney_disease, liver_disease, allergies, current_medicines


class Share(BaseModel):
    doctor_id: int | None = None  # picked from GET /records/doctors
    doctor_email: str | None = Field(default=None, max_length=200)


class Plan(BaseModel):
    diagnosis: str = Field(min_length=2, max_length=300)
    tests: list[str] = []
    medicines: list[str] = []
    advice: str = Field(default="", max_length=4000)
    referral: str | None = Field(default=None, max_length=1000)


def _case(con, case_id: int) -> dict:
    row = con.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No such case.")
    plans = [{"doctor_id": p["doctor_id"], "confirmed": p["confirmed"], **store.unseal(p["data"])}
             for p in con.execute("SELECT * FROM plans WHERE case_id = ? ORDER BY confirmed DESC", (case_id,))]
    shared = [dict(s) for s in con.execute("SELECT u.id, u.name, u.email, s.granted FROM shares s JOIN users u ON u.id = s.doctor_id "
                                           "WHERE s.case_id = ? AND s.revoked IS NULL", (case_id,))]
    return {"id": row["id"], "patient_id": row["patient_id"], "title": row["title"], "created": row["created"], **store.unseal(row["data"]),
            "shared_with": shared, "doctor_plans": plans}


# ---- patient side ----
@router.get("/records/doctors")
def doctors(user: dict = Depends(require_role("patient"))) -> list[dict]:
    """Approved doctors a patient can share with (name, specialty, clinic; never their email or phone)."""
    con = store.connect()
    out = []
    for r in con.execute("SELECT id, name, details FROM users WHERE role = 'doctor' AND status = 'approved' ORDER BY name"):
        d = store.json.loads(r["details"] or "{}")
        out.append({"id": r["id"], "name": r["name"], "specialty": d.get("specialty", ""), "clinic": d.get("clinic", "")})
    con.close()
    return out



@router.post("/records/cases")
def create_case(body: NewCase, user: dict = Depends(require_role("patient"))) -> dict:
    con = store.connect()
    data = body.model_dump(exclude={"title"})
    cur = con.execute("INSERT INTO cases (patient_id, title, data, created) VALUES (?, ?, ?, ?)", (user["id"], body.title, store.seal(data), time.time()))
    store.audit(con, user["id"], "case:create", cur.lastrowid)
    con.commit()
    out = _case(con, cur.lastrowid)
    con.close()
    return out


@router.get("/records/cases")
def my_cases(user: dict = Depends(require_role("patient"))) -> list[dict]:
    con = store.connect()
    ids = [r["id"] for r in con.execute("SELECT id FROM cases WHERE patient_id = ? ORDER BY created DESC", (user["id"],))]
    out = [_case(con, i) for i in ids]
    store.audit(con, user["id"], "case:list")
    con.commit()
    con.close()
    return out


@router.post("/records/cases/{case_id}/share")
def share(case_id: int, body: Share, user: dict = Depends(require_role("patient"))) -> dict:
    con = store.connect()
    case = _case(con, case_id)
    if case["patient_id"] != user["id"]:
        con.close()
        raise HTTPException(status_code=404, detail="No such case.")
    if body.doctor_id is not None:
        doc = con.execute("SELECT * FROM users WHERE id = ? AND role = 'doctor' AND status = 'approved'", (body.doctor_id,)).fetchone()
    else:
        doc = con.execute("SELECT * FROM users WHERE email = ? AND role = 'doctor' AND status = 'approved'", ((body.doctor_email or "").strip().lower(),)).fetchone()
    if not doc:
        con.close()
        raise HTTPException(status_code=404, detail="No approved doctor found. Pick one from the list.")
    con.execute("INSERT OR REPLACE INTO shares (case_id, doctor_id, granted, revoked) VALUES (?, ?, ?, NULL)", (case_id, doc["id"], time.time()))
    store.audit(con, user["id"], f"case:share:{doc['id']}", case_id)
    con.commit()
    out = _case(con, case_id)
    con.close()
    return out


@router.delete("/records/cases/{case_id}/share/{doctor_id}")
def unshare(case_id: int, doctor_id: int, user: dict = Depends(require_role("patient"))) -> dict:
    con = store.connect()
    if _case(con, case_id)["patient_id"] != user["id"]:
        con.close()
        raise HTTPException(status_code=404, detail="No such case.")
    con.execute("UPDATE shares SET revoked = ? WHERE case_id = ? AND doctor_id = ?", (time.time(), case_id, doctor_id))
    store.audit(con, user["id"], f"case:unshare:{doctor_id}", case_id)
    con.commit()
    out = _case(con, case_id)
    con.close()
    return out


# ---- doctor side ----
def _shared(con, case_id: int, doctor_id: int) -> None:
    if not con.execute("SELECT 1 FROM shares WHERE case_id = ? AND doctor_id = ? AND revoked IS NULL", (case_id, doctor_id)).fetchone():
        raise HTTPException(status_code=404, detail="This case is not shared with you.")


@router.get("/doctor/cases")
def doctor_cases(doc: dict = Depends(require_role("doctor"))) -> list[dict]:
    con = store.connect()
    ids = [r["case_id"] for r in con.execute("SELECT case_id FROM shares WHERE doctor_id = ? AND revoked IS NULL ORDER BY granted DESC", (doc["id"],))]
    out = [_case(con, i) for i in ids]
    store.audit(con, doc["id"], "doctor:case:list")
    con.commit()
    con.close()
    return out


@router.get("/doctor/cases/{case_id}")
def doctor_case(case_id: int, doc: dict = Depends(require_role("doctor"))) -> dict:
    con = store.connect()
    _shared(con, case_id, doc["id"])
    store.audit(con, doc["id"], "doctor:case:view", case_id)
    con.commit()
    out = _case(con, case_id)
    con.close()
    return out


@router.post("/doctor/cases/{case_id}/plan")
def confirm_plan(case_id: int, body: Plan, doc: dict = Depends(require_role("doctor"))) -> dict:
    con = store.connect()
    _shared(con, case_id, doc["id"])
    con.execute("INSERT INTO plans (case_id, doctor_id, data, confirmed) VALUES (?, ?, ?, ?)",
                (case_id, doc["id"], store.seal(body.model_dump() | {"doctor_name": doc["name"]}), time.time()))
    store.audit(con, doc["id"], "doctor:plan:confirm", case_id)
    con.commit()
    out = _case(con, case_id)
    con.close()
    return out
