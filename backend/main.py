"""Entry point kept for `uvicorn main:app`. The code lives in api/ (layout: docs/BACKEND_PLAN.md §8)."""
from api.main import app

__all__ = ["app"]
