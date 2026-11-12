"""Reduced kidney function (eGFR below 60) from urea and uric acid, without creatinine. Verified on NHANES 2021-23 (sensitivity 0.839, AUC 0.917). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_low_egfr_labs"
    target = "low_egfr_labs"
    input_schema = schema_for("low_egfr_labs")
    high = "Possible reduced kidney function"
    low = "Reduced kidney function less likely"
    note = "A creatinine blood test (eGFR) confirms it."
