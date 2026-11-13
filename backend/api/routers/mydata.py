"""GET /me/data: everything Curanex saved for the logged-in user (chats, uploads with temporary links, reports, model results,
cases, doctor plans). Only the user's own rows are returned."""
from fastapi import APIRouter, Depends

from api import cloud
from api.auth.deps import current_user

router = APIRouter(tags=["my data"])


@router.get("/me/data")
def my_data(user: dict = Depends(current_user)) -> dict:
    uid = user["id"]
    uploads = cloud.rows("uploads", uid, 50)
    for u in uploads:
        u["url"] = cloud.signed_url(u.get("file_path") or "")
    reports = cloud.rows("reports", uid, 50)
    for r in reports:
        r["url"] = cloud.signed_url(r.get("file_path") or "")
    return {"cloud": cloud.enabled(), "chats": cloud.rows("chat_messages", uid, 200), "uploads": uploads, "reports": reports,
            "model_results": cloud.rows("model_results", uid, 100), "cases": cloud.rows("cases", uid, 50), "vitals": cloud.rows("vitals", uid, 20), "meals": cloud.rows("meals", uid, 50),
            "doctor_plans": cloud.rows("doctor_plans", uid, 50, user_column="patient_id")}
