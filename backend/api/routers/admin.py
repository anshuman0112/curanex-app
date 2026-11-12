"""Admin (the owner): approve, reject or suspend doctors by hand, and read the audit log."""

import time

from fastapi import APIRouter, Depends, HTTPException

from api.auth import store
from api.auth.deps import require_role

router = APIRouter(prefix="/admin", tags=["admin"])
ACTIONS = {"approve": "approved", "reject": "rejected", "suspend": "suspended"}


@router.get("/doctors")
def doctors(admin: dict = Depends(require_role("admin"))) -> list[dict]:
    con = store.connect()
    rows = [store.public(dict(r)) | {"created": r["created"]} for r in con.execute("SELECT * FROM users WHERE role = 'doctor' ORDER BY status, created DESC")]
    con.close()
    return rows


@router.post("/doctors/{doctor_id}/{action}")
def decide(doctor_id: int, action: str, admin: dict = Depends(require_role("admin"))) -> dict:
    if action not in ACTIONS:
        raise HTTPException(status_code=404, detail="Unknown action.")
    con = store.connect()
    row = con.execute("SELECT * FROM users WHERE id = ? AND role = 'doctor'", (doctor_id,)).fetchone()
    if not row:
        con.close()
        raise HTTPException(status_code=404, detail="No such doctor.")
    con.execute("UPDATE users SET status = ? WHERE id = ?", (ACTIONS[action], doctor_id))
    if action != "approve":
        con.execute("DELETE FROM tokens WHERE user_id = ?", (doctor_id,))
    store.audit(con, admin["id"], f"doctor:{action}:{doctor_id}")
    con.commit()
    out = store.public(dict(con.execute("SELECT * FROM users WHERE id = ?", (doctor_id,)).fetchone()))
    con.close()
    return out


@router.get("/audit")
def audit(limit: int = 200, admin: dict = Depends(require_role("admin"))) -> list[dict]:
    con = store.connect()
    rows = [{"user_id": r["user_id"], "action": r["action"], "case_id": r["case_id"], "at": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["at"]))}
            for r in con.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (min(limit, 1000),))]
    con.close()
    return rows
