"""Insulin resistance (fasting HOMA-IR > 2.5) from height, weight and simple questions, no blood test. Verified on NHANES 2021-23 (sensitivity 0.884, AUC 0.800). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_insulin_resistance_questions"
    target = "insulin_resistance_questions"
    input_schema = schema_for("insulin_resistance_questions")
    high = "Possible insulin resistance"
    low = "Insulin resistance less likely"
    note = "A fasting insulin and glucose test confirms it."
