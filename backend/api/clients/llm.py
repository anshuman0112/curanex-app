"""LLM client: OpenAI-compatible /chat/completions with tool calling (Groq, Gemini, OpenRouter, Ollama, AI Gateway).
Keys come from the server environment only. Returns None when no key is set or the provider fails,
so callers can fall back without breaking the app."""
import httpx

from api.core.config import get_settings
from api.core.logging import log


def llm_configured() -> bool:
    base, key = get_settings().llm_endpoint
    return bool(key) or "localhost" in base or "ollama" in base


def chat_completion(messages: list[dict], tools: list[dict] | None = None) -> dict | None:
    """One completion. Returns the assistant message dict ({content, tool_calls?}) or None."""
    s = get_settings()
    if not llm_configured():
        return None
    base, key = s.llm_endpoint
    body: dict = {"model": s.llm_model, "messages": messages, "max_tokens": s.llm_max_tokens, "temperature": 0.3}
    if "gpt-oss" in s.llm_model and s.llm_reasoning_effort:
        body["reasoning_effort"] = s.llm_reasoning_effort
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    try:
        r = httpx.post(f"{base.rstrip('/')}/chat/completions", json=body, timeout=s.llm_timeout_s,
                       headers={"Authorization": f"Bearer {key}"} if key else {})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]
    except Exception as e:  # keep the app usable if the provider fails
        status = getattr(getattr(e, "response", None), "status_code", None)
        log.warning("llm_error", error=type(e).__name__, status=status)
        return None


def call_llm(messages: list[dict], system: str) -> str | None:
    msg = chat_completion([{"role": "system", "content": system}, *messages])
    return (msg or {}).get("content") or None
