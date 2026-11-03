"""LLM provider behind one function. Anthropic Messages API, direct or via Vercel AI Gateway; keys from the server env only."""
from api.core.config import Settings, get_settings
from api.core.logging import log


def llm_configured(s: Settings) -> bool:
    return bool(s.ai_gateway_api_key or s.llm_api_key)


def make_client(s: Settings):
    import anthropic

    if s.ai_gateway_api_key:
        return anthropic.Anthropic(api_key=s.ai_gateway_api_key, base_url=s.ai_gateway_base_url)
    return anthropic.Anthropic(api_key=s.llm_api_key)


def call_llm(messages: list[dict], system: str) -> str | None:
    s = get_settings()
    if not llm_configured(s):
        return None
    try:
        client = make_client(s)
        r = client.messages.create(model=s.llm_model, max_tokens=s.llm_max_tokens, system=system, messages=messages)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    except Exception as e:  # keep the app usable if the provider fails
        log.warning("llm_error", error=type(e).__name__)
        return None
