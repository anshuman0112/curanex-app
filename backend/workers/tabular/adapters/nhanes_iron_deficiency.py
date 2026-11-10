"""Iron deficiency check from a complete blood count. Trained by the Curanex team (training/train_nhanes.py) on US CDC
NHANES, with iron stores measured by ferritin."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_iron_deficiency"
    target = "iron_deficiency"
    input_schema = schema_for("iron_deficiency")
    high = "Possible low iron stores"
    low = "Low iron stores less likely"
    note = "A serum ferritin blood test confirms low iron. Don't start iron tablets without a doctor's advice."

    def validate(self, inputs: dict) -> list[str]:
        errors = super().validate(inputs)
        if errors:
            return errors
        # Verified only on women and girls aged 12-49 (the only group with ferritin in the 2021-23 test cycle).
        if str(inputs.get("sex") or "").strip().lower()[:1] != "f" or not 12 <= float(inputs["age"]) <= 49:
            return ["sex: this check was tested only on women and girls aged 12 to 49"]
        return []
