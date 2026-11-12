"""Insulin resistance (fasting HOMA-IR > 2.5) from a routine blood test without glucose or insulin. Verified on NHANES 2021-23 (sensitivity 0.854, AUC 0.834). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_insulin_resistance_labs"
    target = "insulin_resistance_labs"
    input_schema = schema_for("insulin_resistance_labs")
    high = "Possible insulin resistance"
    low = "Insulin resistance less likely"
    note = "A fasting insulin and glucose test confirms it."
