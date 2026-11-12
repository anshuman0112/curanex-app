"""Diabetes (told, or HbA1c >= 6.5%) from a routine blood test without sugar tests. Verified on NHANES 2021-23 (sensitivity 0.885, AUC 0.812). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_diabetes_any_labs"
    target = "diabetes_any_labs"
    input_schema = schema_for("diabetes_any_labs")
    high = "Possible diabetes"
    low = "Diabetes less likely"
    note = "An HbA1c or fasting blood sugar test confirms diabetes."
