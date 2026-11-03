"""Orchestrator. Phase B2 adds model search + the tool loop (ARCHITECTURE.md §2-§5) here.
For now: one LLM call with the user context, then the safety post-check."""
import json

from api.clients.llm import call_llm
from api.orchestrator.prompts import SYSTEM, VOICE
from api.safety import post_check
from api.schemas.chat import ChatRequest

NO_LLM = ("The AI model isn't configured on the server yet (set LLM_API_KEY). "
          "The app, safety checks and SOS work; answers will appear once the key is added.")


def orchestrate(req: ChatRequest) -> list[dict]:
    system = SYSTEM + (VOICE if req.via == "voice" else "")
    system += "\n\nUser context:\n" + json.dumps(req.context or {}, ensure_ascii=False)[:4000]
    msgs = [t.model_dump() for t in req.history[-10:]] + [{"role": "user", "content": req.message or "(user sent inputs)"}]
    text = call_llm(msgs, system) or NO_LLM
    return [{"type": "text", "text": post_check(text)}]
