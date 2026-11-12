"""Raised inflammation (hs-CRP > 3 mg/L) from a blood count, albumin and lipids. Verified on NHANES 2021-23 (sensitivity 0.842, AUC 0.807). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_inflammation_crp_labs"
    target = "inflammation_crp_labs"
    input_schema = schema_for("inflammation_crp_labs")
    high = "Possible raised inflammation"
    low = "Raised inflammation less likely"
    note = "A high-sensitivity CRP test confirms it."
