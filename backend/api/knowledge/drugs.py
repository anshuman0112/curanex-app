"""Medicines for the doctor module. Doses are NEVER written by the AI: the dose text is copied word for word from the official
drug label in the library, together with the label sections that matter for this patient (pregnancy, children, older adults,
kidney or liver disease, allergies). If the medicine has no label in the library, no dose is shown.
"""

from api.knowledge.search import drug_sections

ALWAYS = ("boxed_warning", "dosage_and_administration", "contraindications", "directions", "do_not_use")


def patient_fields(patient: dict) -> tuple[str, ...]:
    """Extra label sections to show for this patient."""
    extra = ["drug_interactions"] if patient.get("current_medicines") else []
    age = patient.get("age")
    if patient.get("pregnant"):
        extra.append("pregnancy")
    if isinstance(age, (int, float)) and age < 18:
        extra.append("pediatric_use")
    if isinstance(age, (int, float)) and age >= 65:
        extra.append("geriatric_use")
    if patient.get("kidney_disease") or patient.get("liver_disease"):
        extra.append("use_in_specific_populations")
    extra.append("warnings_and_cautions")
    return tuple(dict.fromkeys(extra))


def medicine_card(name: str, patient: dict) -> dict:
    fields = ALWAYS + patient_fields(patient)
    sections = drug_sections(name, fields)
    if not sections:
        return {"medicine": name, "found": False, "note": "No official drug label for this medicine in the library, so no dose is shown."}
    first = sections[0]
    # An allergy matches if it is the medicine itself or is named in the label's contraindications (e.g. penicillin -> amoxicillin)
    contra = " ".join(s["text"] for s in sections if s.get("field") in ("contraindications", "boxed_warning", "do_not_use")).lower()
    allergy = any(a and (a.lower() in name.lower() or a.lower().rstrip("s") in contra) for a in (patient.get("allergies") or []))
    return {
        "medicine": name, "found": True, "label_title": first["title"], "label_url": first["url"], "licence": first["licence"],
        "label_date": first.get("effective", ""),
        "sections": [{"section": s["section"], "text": s["text"]} for s in sections],
        "warnings": (["The patient reports an allergy matching this medicine."] if allergy else []),
        "note": "Dose and warnings are quoted word for word from the official label (US FDA). Check the Indian product's leaflet and adjust to the patient.",
    }
