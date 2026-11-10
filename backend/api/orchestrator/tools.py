"""Turns registry entries into LLM tools and /chat blocks (ARCHITECTURE.md §5-§6)."""
import re

IMAGE_MODALITIES = {"phone_photo", "medical_image_upload", "image"}
UPLOAD_KINDS = [
    "skin_photo", "dental_photo", "chest_xray", "knee_xray", "bone_xray", "brain_mri",
    "blood_cell", "endoscopy", "ct_scan", "fundus", "report", "ultrasound",
]
# registry from_profile key -> key in the app's getUserContext().profile
PROFILE_KEYS = {"age": "age", "sex": "sex", "bmi": "bmi", "height": "heightCm", "height_cm": "heightCm",
                "weight": "weightKg", "weight_kg": "weightKg"}
# "has_<condition>" form keys answered "Yes" from the conditions already in the profile
CONDITION_WORDS = {
    "diabetes": r"diabet|sugar", "heart_disease": r"heart|cardiac|angina|coronary", "stroke": r"stroke|paralys",
    "asthma": r"asthma", "copd": r"copd|bronchitis|emphysema", "kidney_disease": r"kidney|renal|ckd", "arthritis": r"arthritis",
    "depression": r"depress", "high_blood_pressure": r"blood pressure|hypertension|\bbp\b", "high_cholesterol": r"cholesterol|lipid",
}
DISCLAIMER = "Not a diagnosis"
MAX_FORM_FIELDS = 5


def is_image_model(entry: dict) -> bool:
    return entry.get("modality") in IMAGE_MODALITIES


def tool_name(entry: dict) -> str:
    return ("run_" + re.sub(r"[^a-zA-Z0-9_]", "_", entry["id"]))[:64]


def model_name(entry: dict) -> str:
    return entry.get("name") or ", ".join(entry.get("tasks") or []) or entry["id"]


def _clean_schema(schema: dict) -> dict:
    """JSON Schema the providers accept: drop our own keys (label, unit, from_profile) into descriptions."""
    props = {}
    for key, spec in (schema.get("properties") or {}).items():
        p = {k: spec[k] for k in ("type", "minimum", "maximum", "enum") if k in spec}
        p.setdefault("type", "string")
        desc = " ".join(x for x in (spec.get("label"), f"({spec['unit']})" if spec.get("unit") else None,
                                    "pre-filled from the user's profile if known" if spec.get("from_profile") else None) if x)
        if desc:
            p["description"] = desc
        props[key] = p
    return {"type": "object", "properties": props, "required": list(schema.get("required") or [])}


def model_tool(entry: dict) -> dict:
    desc = f"{model_name(entry)}. {entry.get('description', '')} Limitations: {entry.get('limitations', '')}"
    if is_image_model(entry):
        desc += " Needs an image: call request_image first unless the user already uploaded one."
    return {"type": "function", "function": {"name": tool_name(entry), "description": desc[:1000],
                                             "parameters": _clean_schema(entry.get("input_schema") or {})}}


FIXED_TOOLS = [
    {"type": "function", "function": {
        "name": "request_user_input",
        "description": "Ask the user for missing model inputs. The app shows a short form. Ends your turn.",
        "parameters": {"type": "object", "required": ["model_id", "fields"], "properties": {
            "model_id": {"type": "string"}, "fields": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string", "description": "One friendly sentence shown above the form"}}}}},
    {"type": "function", "function": {
        "name": "request_image",
        "description": "Ask the user to take or upload a photo or medical image for a model. Ends your turn.",
        "parameters": {"type": "object", "required": ["model_id", "kind"], "properties": {
            "model_id": {"type": "string"}, "kind": {"type": "string", "enum": UPLOAD_KINDS},
            "instructions": {"type": "string"}}}}},
]


def prefill(entry: dict, profile: dict | None) -> dict:
    out = {}
    known = " ".join(str(c) for c in (profile or {}).get("conditions") or []).lower()
    for key, spec in ((entry.get("input_schema") or {}).get("properties") or {}).items():
        words = CONDITION_WORDS.get(key.removeprefix("has_")) if key.startswith("has_") else None
        if words and known and re.search(words, known):
            out[key] = "Yes"
        if spec.get("from_profile") and profile:
            v = profile.get(PROFILE_KEYS.get(key, key))
            if v not in (None, ""):
                out[key] = v
    return out


def coerce(entry: dict, values: dict) -> dict:
    props = (entry.get("input_schema") or {}).get("properties") or {}
    out = {}
    for k, v in values.items():
        t = (props.get(k) or {}).get("type")
        try:
            out[k] = int(float(v)) if t == "integer" else float(v) if t == "number" else v
        except (TypeError, ValueError):
            out[k] = v
    return out


def missing_fields(entry: dict, inputs: dict) -> list[str]:
    return [k for k in (entry.get("input_schema") or {}).get("required", []) if inputs.get(k) in (None, "")]


def form_fields(entry: dict, missing: list[str], prefilled: dict) -> list[str]:
    """Missing required fields, then optional ones worth asking, at most MAX_FORM_FIELDS."""
    props = (entry.get("input_schema") or {}).get("properties") or {}
    optional = [k for k, v in props.items() if k not in missing and k not in prefilled and not v.get("from_profile")]
    return (missing + optional)[:MAX_FORM_FIELDS]


def form_request_block(entry: dict, fields: list[str], prefilled: dict, defaults: dict | None = None) -> dict:
    props = (entry.get("input_schema") or {}).get("properties") or {}
    required = set((entry.get("input_schema") or {}).get("required") or [])
    return {
        "type": "form_request", "model_id": entry["id"], "title": model_name(entry),
        "fields": [{"key": k, "label": props.get(k, {}).get("label", k.replace("_", " ").capitalize()),
                    "unit": props.get(k, {}).get("unit"),
                    "type": "number" if props.get(k, {}).get("type") in ("number", "integer") else "text",
                    "optional": k not in required, **({"options": props[k]["enum"]} if props.get(k, {}).get("enum") else {})}
                   for k in fields if k in props],
        "prefilled": prefilled,
        **({"defaults": defaults} if defaults else {}),
    }


def upload_kind(entry: dict, requested: str | None = None) -> str:
    if requested in UPLOAD_KINDS:
        return requested
    return entry.get("upload_kind") or ("skin_photo" if entry.get("modality") == "phone_photo" else "chest_xray")


def upload_request_block(entry: dict, kind: str | None = None, instructions: str | None = None) -> dict:
    return {"type": "upload_request", "model_id": entry["id"], "kind": upload_kind(entry, kind),
            "instructions": instructions or entry.get("upload_instructions") or "Use a clear, well-lit image."}


def model_result_block(entry: dict, pred: dict) -> dict:
    explanation = dict(pred.get("explanation") or {})
    heatmap = explanation.pop("heatmap_png_b64", None)
    return {
        "type": "model_result", "model_id": entry["id"], "model_name": model_name(entry),
        "label": pred["label"], "probabilities": pred["probabilities"],
        "trained_on": entry.get("trained_on") or "Not documented",
        "limitations": entry.get("limitations") or "Not documented",
        "credit": entry.get("credit") or f"Model from {entry.get('repo', 'unknown')}",
        "verified": bool(entry.get("verified")), "metrics": entry.get("metrics"),
        "explanation": explanation, "warnings": pred.get("warnings") or [],
        "heatmap_url": f"data:image/png;base64,{heatmap}" if heatmap else None,
        "disclaimer": DISCLAIMER,
    }
