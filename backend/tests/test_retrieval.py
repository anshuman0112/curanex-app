"""WORKFLOW.md A9: model search on real phrases. A wrong model firing is worse than no model."""
import importlib.util
import json
from pathlib import Path

import pytest

from api.core.config import get_settings
from api.orchestrator import retriever
from api.registry import routable_models

CASES = json.loads((Path(__file__).parent / "retrieval_cases.json").read_text(encoding="utf-8"))
needs_embedder = pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None,
                                    reason="sentence-transformers not installed")


@pytest.fixture
def search(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_DUMMY_MODEL", "false")
    monkeypatch.setenv("PREVIEW_MODELS", "mlmed_torchxrayvision")  # tested before it is switched on
    monkeypatch.setenv("EMBEDDINGS_CACHE", str(tmp_path / "emb.npz"))
    get_settings.cache_clear()
    retriever.set_embedder(None)
    yield retriever.search
    retriever.set_embedder(None)
    get_settings.cache_clear()


@needs_embedder
def test_routing_on_real_phrases(search):
    integrated = {m["id"] for m in routable_models()}
    cases = [c for c in CASES if c["expect"] is None or c["expect"] in integrated]
    results = [(c, [m["id"] for m, _ in search(c["text"])]) for c in cases]
    right = [c for c, found in results if (c["expect"] in found[:1] if c["expect"] else not found)]
    wrong_fires = [c["text"] for c, found in results if c["expect"] is None and found]
    assert len(right) / len(cases) >= 0.85, [c["text"] for c, _ in results if c not in right]
    assert len(wrong_fires) <= 0.05 * len(cases) + 1e-9, wrong_fires
