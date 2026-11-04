"""Orchestrator (WORKFLOW.md B6-B20): model search -> LLM tool loop -> worker -> explanation -> blocks.

Paths:
- Follow-up with model_id + form_values/image_id: run that model directly, then explain.
- No model above the search threshold: the LLM answers directly.
- Candidates found: the LLM gets one tool per candidate plus request_user_input / request_image (max N calls).
- No LLM available: deterministic fallback asks for the top candidate's inputs, and results are explained
  from a fixed template, so the model flow still works.
The safety post-check runs on every text block."""
import base64
import json

from api.clients import workers
from api.clients.llm import call_llm, chat_completion, llm_configured
from api.core.config import get_settings
from api.core.logging import log
from api.orchestrator import tools as T
from api.orchestrator.prompts import EXPLAIN, SYSTEM, TOOLS, VOICE
from api.orchestrator.retriever import search
from api.registry import get_routable
from api.safety import post_check
from api.schemas.chat import ChatRequest
from api.uploads_store import load_image

NO_LLM = ("The AI model isn't configured on the server yet (set LLM_API_KEY). "
          "The app, safety checks and SOS work; answers will appear once the key is added.")


def _text(text: str) -> dict:
    return {"type": "text", "text": post_check(text)}


def _profile(req: ChatRequest) -> dict:
    return (req.context or {}).get("profile") or {}


def _system(req: ChatRequest, extra: str = "") -> str:
    s = SYSTEM + extra + (VOICE if req.via == "voice" else "")
    return s + "\n\nUser context:\n" + json.dumps(req.context or {}, ensure_ascii=False)[:4000]


def _history(req: ChatRequest) -> list[dict]:
    return [t.model_dump() for t in req.history[-10:]]


def orchestrate(req: ChatRequest) -> list[dict]:
    if req.model_id and (req.form_values or req.image_id):
        entry = get_routable(req.model_id)
        if entry is None:
            return [_text("That screening model isn't available right now. Please describe your symptoms instead.")]
        return run_model(entry, req, {k: v for k, v in (req.form_values or {}).items() if v not in (None, "")})

    recent = " ".join(t.content[:300] for t in req.history[-2:])
    candidates = search(req.message, context=recent)
    log.info("model_search", candidates=[(m["id"], round(score, 3)) for m, score in candidates])
    if not candidates:
        return answer_directly(req)
    if not llm_configured():
        return ask_for_inputs(candidates[0][0], req)
    return tool_loop(req, [m for m, _ in candidates])


def answer_directly(req: ChatRequest) -> list[dict]:
    msgs = _history(req) + [{"role": "user", "content": req.message or "(user sent inputs)"}]
    return [_text(call_llm(msgs, _system(req)) or NO_LLM)]


def ask_for_inputs(entry: dict, req: ChatRequest, reason: str | None = None) -> list[dict]:
    """Form or upload card for a model, with profile values pre-filled."""
    if T.is_image_model(entry):
        return [_text(reason or f"I can run a screening check ({T.model_name(entry)}) if you upload an image."),
                T.upload_request_block(entry)]
    pre = T.prefill(entry, _profile(req))
    missing = T.missing_fields(entry, pre) or [k for k in (entry["input_schema"].get("required") or []) if k not in pre]
    return [_text(reason or f"I can run a quick screening check ({T.model_name(entry)}). I need a few details."),
            T.form_request_block(entry, T.form_fields(entry, missing, pre), pre)]


def _predict(entry: dict, req: ChatRequest, inputs: dict) -> dict | list[dict]:
    """Prediction dict, or blocks to return instead (missing inputs / errors)."""
    inputs = T.coerce(entry, {**T.prefill(entry, _profile(req)), **inputs})
    if T.is_image_model(entry):
        image = load_image(req.image_id or "")
        if image is None:
            return ask_for_inputs(entry, req)
        inputs["image_b64"] = base64.b64encode(image).decode()
    else:
        missing = T.missing_fields(entry, inputs)
        if missing:
            pre = T.prefill(entry, _profile(req))
            return [_text("I still need a few details to run this check."),
                    T.form_request_block(entry, T.form_fields(entry, missing, pre), pre)]
    try:
        return workers.predict(entry.get("worker", "tabular"), entry["id"], inputs)
    except workers.WorkerError as e:
        log.warning("worker_error", model_id=entry["id"], status=e.status)
        if e.status == 422:
            return [_text(f"I couldn't use those inputs: {e}. Please check them and try again.")]
        return [_text("The screening model is unavailable right now. Please try again in a minute, "
                      "or describe your symptoms and I'll help without it.")]


def explain(entry: dict, block: dict, req: ChatRequest) -> str:
    result = {k: block[k] for k in ("model_name", "label", "probabilities", "trained_on", "limitations", "verified")}
    msgs = _history(req) + [{"role": "user", "content": EXPLAIN.format(result=json.dumps(result))}]
    text = call_llm(msgs, _system(req)) if llm_configured() else None
    if text:
        return text
    pct = round(100 * max(block["probabilities"].values()))
    return (f"The {block['model_name']} model's estimate is \"{block['label']}\" ({pct}% confidence). "
            f"This is a screening estimate, not a diagnosis. It was trained on {block['trained_on']}. "
            f"Limitations: {block['limitations']} Please discuss the result with a doctor.")


def run_model(entry: dict, req: ChatRequest, inputs: dict) -> list[dict]:
    pred = _predict(entry, req, inputs)
    if isinstance(pred, list):
        return pred
    block = T.model_result_block(entry, pred)
    return [block, _text(explain(entry, block, req))]


def tool_loop(req: ChatRequest, candidates: list[dict]) -> list[dict]:
    by_tool = {T.tool_name(m): m for m in candidates}
    by_id = {m["id"]: m for m in candidates}
    tool_defs = [T.model_tool(m) for m in candidates] + T.FIXED_TOOLS
    messages = [{"role": "system", "content": _system(req, TOOLS)}, *_history(req),
                {"role": "user", "content": req.message}]
    blocks: list[dict] = []
    for _ in range(get_settings().max_tool_calls):
        msg = chat_completion(messages, tool_defs)
        if msg is None:
            return blocks + (ask_for_inputs(candidates[0], req) if not blocks else [])
        calls = msg.get("tool_calls") or []
        if not calls:
            return blocks + [_text(msg.get("content") or "Could you tell me a bit more?")]
        call = calls[0]
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if name in ("request_user_input", "request_image"):
            entry = by_id.get(args.get("model_id")) or candidates[0]
            if name == "request_image" or T.is_image_model(entry):
                return [_text(msg.get("content") or f"Please upload an image for the {T.model_name(entry)} check."),
                        T.upload_request_block(entry, args.get("kind"), args.get("instructions"))]
            return ask_for_inputs(entry, req, args.get("reason") or msg.get("content"))
        entry = by_tool.get(name)
        if entry is None:
            return blocks + [_text(msg.get("content") or "Could you tell me a bit more?")]
        if T.is_image_model(entry) and not req.image_id:
            return ask_for_inputs(entry, req, msg.get("content"))
        pred = _predict(entry, req, {k: v for k, v in args.items() if v not in (None, "")})
        if isinstance(pred, list):
            return pred
        block = T.model_result_block(entry, pred)
        blocks.append(block)
        messages += [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": [call]},
                     {"role": "tool", "tool_call_id": call.get("id", "call_0"),
                      "content": json.dumps({k: block[k] for k in ("label", "probabilities", "trained_on", "limitations")})}]
    if blocks:
        return blocks + [_text(explain(by_id[blocks[-1]["model_id"]], blocks[-1], req))]
    return [_text("Could you tell me a bit more about what's going on?")]
