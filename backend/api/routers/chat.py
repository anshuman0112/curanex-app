from fastapi import APIRouter, Depends

from api import cloud
from api.auth.deps import optional_user
from api.orchestrator.loop import orchestrate
from api.safety import red_flag
from api.schemas.chat import ChatRequest

router = APIRouter()


def _remember(user: dict | None, req: ChatRequest, blocks: list[dict]) -> None:
    """Logged-in patients: every turn, the full reply and any model result go to their own Supabase rows."""
    if not user:
        return
    who = {"user_id": user["id"], "user_email": user["email"]}
    said = req.message or ("(submitted a form)" if req.form_values else "(uploaded an image)" if req.image_id else "(report)" if req.report_text else "")
    cloud.save("chat_messages", who | {"session_id": req.session_id, "role": "user", "content": said,
                                       "blocks": {"form_values": req.form_values, "image_id": req.image_id, "model_id": req.model_id} if (req.form_values or req.image_id) else None})
    text = " ".join(str(b.get("text") or b.get("content") or b.get("summary") or "") for b in blocks).strip()
    cloud.save("chat_messages", who | {"session_id": req.session_id, "role": "assistant", "content": text[:8000], "blocks": blocks})
    for b in blocks:
        if b.get("type") == "model_result":
            cloud.save("model_results", who | {"session_id": req.session_id, "model_id": b.get("model_id"), "model_name": b.get("model_name"),
                                               "label": b.get("label"), "flag": b.get("flag"), "probabilities": b.get("probabilities"),
                                               "inputs": req.form_values, "verified": b.get("verified")})


@router.post("/chat")
@router.post("/v1/chat")
def chat(req: ChatRequest, user: dict | None = Depends(optional_user)):
    reason = red_flag(req.message or "")
    blocks = [{"type": "escalate", "reason": reason}] if reason else orchestrate(req)
    _remember(user, req, blocks)
    return {"blocks": blocks}
