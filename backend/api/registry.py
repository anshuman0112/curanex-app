"""Model registry loader. Only entries with integrated=true are routable (offered to model search and the LLM)."""
import json
from functools import lru_cache
from pathlib import Path

from api.core.config import get_settings

TEST_MODELS = Path(__file__).resolve().parents[1] / "registry" / "test_models.json"


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> tuple[dict, ...]:
    return tuple(json.loads(Path(path).read_text(encoding="utf-8")).get("models", []))


def all_models() -> list[dict]:
    path = get_settings().registry_path
    return list(_load(str(path), path.stat().st_mtime)) if path.exists() else []


def routable_models() -> list[dict]:
    preview = {i.strip() for i in get_settings().preview_models.split(",") if i.strip()}
    models = [m for m in all_models() if m.get("integrated") or m["id"] in preview]
    if get_settings().enable_dummy_model:
        models += list(_load(str(TEST_MODELS), TEST_MODELS.stat().st_mtime))
    return models


def get_routable(model_id: str) -> dict | None:
    return next((m for m in routable_models() if m["id"] == model_id), None)
