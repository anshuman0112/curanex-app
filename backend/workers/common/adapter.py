"""Adapter contract shared by every worker (ARCHITECTURE.md §4, BACKEND_PLAN §3.1)."""
from dataclasses import dataclass, field


@dataclass
class Prediction:
    model_id: str
    label: str                       # top class, plain language
    probabilities: dict[str, float]  # all classes
    explanation: dict = field(default_factory=dict)  # e.g. {"top_factors": [...]} or {"heatmap_png_b64": ...}
    warnings: list[str] = field(default_factory=list)


class ModelAdapter:
    id: str = ""
    input_schema: dict = {}

    def load(self) -> None:
        """Called once, lazily, before the first prediction."""

    def validate(self, inputs: dict) -> list[str]:
        """Missing or invalid fields. Default: check against input_schema."""
        return check_schema(self.input_schema, inputs)

    def predict(self, inputs: dict) -> Prediction:
        raise NotImplementedError


def check_schema(schema: dict, inputs: dict) -> list[str]:
    errors = []
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if inputs.get(key) in (None, ""):
            errors.append(f"{key}: missing")
    for key, spec in props.items():
        if inputs.get(key) not in (None, "") and "enum" in spec and inputs[key] not in spec["enum"]:
            errors.append(f"{key}: not one of the choices")
        if inputs.get(key) in (None, "") or spec.get("type") not in ("number", "integer"):
            continue
        try:
            v = float(inputs[key])
        except (TypeError, ValueError):
            errors.append(f"{key}: not a number")
            continue
        if "minimum" in spec and v < spec["minimum"] or "maximum" in spec and v > spec["maximum"]:
            errors.append(f"{key}: out of range {spec.get('minimum')}–{spec.get('maximum')}")
    return errors
