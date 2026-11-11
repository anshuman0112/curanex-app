"""Kidney damage (eGFR < 60 or urine albumin/creatinine >= 30) from a routine blood test without creatinine or urine tests.
Trained by the Curanex team on US CDC NHANES 2017-18; Verified on NHANES 2021-23 (sensitivity 0.845, AUC 0.800)."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_kidney_damage_labs"
    target = "kidney_damage_labs"
    input_schema = schema_for("kidney_damage_labs")
    high = "Possible kidney damage"
    low = "Kidney damage less likely"
    note = "A creatinine (eGFR) blood test and a urine albumin test confirm kidney damage."
