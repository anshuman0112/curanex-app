"""Traffic-light flag on model results: green only when the model found nothing; never red (emergencies are rule-based)."""

import json
from pathlib import Path

from api.orchestrator.tools import model_result_block, result_flag

REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "model_registry.json"


def test_clear_results_are_green():
    for label in ("Lower risk", "Diabetes less likely", "No clear signs of AMD", "No parasite signs", "No clear abnormality found"):
        assert result_flag(label) == "green", label


def test_findings_are_amber():
    for label in ("Higher risk", "Possible diabetes", "Tooth decay signs", "Malaria parasite signs", "Looks more like a benign lump", "Something may be abnormal"):
        assert result_flag(label) == "amber", label


def test_every_live_label_gets_a_flag():
    models = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    for m in models:
        if m.get("integrated"):
            for label in m.get("output_labels") or []:
                assert result_flag(label) in ("green", "amber")


def test_block_carries_flag():
    entry = {"id": "x", "name": "X check"}
    block = model_result_block(entry, {"label": "Lower risk", "probabilities": {"Lower risk": 0.9, "Higher risk": 0.1}})
    assert block["flag"] == "green"
