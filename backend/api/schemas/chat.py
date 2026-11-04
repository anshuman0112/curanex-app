"""/chat request schema (ARCHITECTURE.md §6, WORKFLOW.md B4)."""
from typing import Any, Literal

from pydantic import BaseModel


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = ""
    via: Literal["text", "voice"] = "text"
    context: dict[str, Any] | None = None
    history: list[Turn] = []
    form_values: dict[str, Any] | None = None
    image_id: str | None = None
    model_id: str | None = None
    report_text: str | None = None
