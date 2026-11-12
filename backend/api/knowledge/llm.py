"""LLM calls for the doctor module only: temperature 0, JSON output, same provider settings as the app.
Kept separate from api/clients/llm.py so the doctor module never changes how the patient app behaves."""

import json
import time

import httpx

from api.clients.llm import llm_configured
from api.core.config import get_settings
from api.core.logging import log


ATTEMPTS = 6
MAX_WAIT_S = 30


def ask_json(system: str, user: str, max_tokens: int = 1800) -> dict | None:
    """One deterministic JSON completion. Returns the parsed object, or None if the provider fails or the reply isn't JSON."""
    s = get_settings()
    if not llm_configured():
        return None
    base, key = s.llm_endpoint
    body: dict = {"model": s.llm_model, "temperature": 0, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if "gpt-oss" in s.llm_model and s.llm_reasoning_effort:
        body["reasoning_effort"] = s.llm_reasoning_effort
    for attempt in range(ATTEMPTS):
        wait = 3.0 * (attempt + 1)
        try:
            r = httpx.post(f"{base.rstrip('/')}/chat/completions", json=body, timeout=s.llm_timeout_s,
                           headers={"Authorization": f"Bearer {key}"} if key else {})
            r.raise_for_status()
            return json.loads(r.json()["choices"][0]["message"].get("content") or "")
        except (json.JSONDecodeError, KeyError):
            log.warning("doctor_llm_not_json")
        except Exception as e:
            response = getattr(e, "response", None)
            status = getattr(response, "status_code", None)
            log.warning("doctor_llm_error", error=type(e).__name__, status=status)
            if status not in (400, 429):
                return None
            after = response.headers.get("retry-after", "") if response is not None else ""
            if after.replace(".", "", 1).isdigit():  # the free tier says how long to wait; respect it
                wait = float(after) + 1
        time.sleep(min(wait, MAX_WAIT_S))
    return None
