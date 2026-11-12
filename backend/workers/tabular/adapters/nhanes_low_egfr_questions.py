"""Reduced kidney function (eGFR < 60) from age, weight and simple questions, no blood test. Verified on NHANES 2021-23 (sensitivity 0.824, AUC 0.853). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_low_egfr_questions"
    target = "low_egfr_questions"
    input_schema = schema_for("low_egfr_questions")
    high = "Possible reduced kidney function"
    low = "Reduced kidney function less likely"
    note = "A creatinine blood test (eGFR) confirms it."
