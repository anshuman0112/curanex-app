from fastapi import APIRouter

from api.orchestrator.loop import orchestrate
from api.safety import red_flag
from api.schemas.chat import ChatRequest

router = APIRouter()


@router.post("/chat")
@router.post("/v1/chat")
def chat(req: ChatRequest):
    reason = red_flag(req.message or "")
    if reason:
        return {"blocks": [{"type": "escalate", "reason": reason}]}
    return {"blocks": orchestrate(req)}
