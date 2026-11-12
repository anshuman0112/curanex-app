"""Chronic kidney disease from a hospital report, trained by the Curanex team on UCI Chronic Kidney Disease (Apollo Hospitals,
Tamil Nadu, India); Verified on Enam Medical College, Bangladesh (2023): sensitivity 0.914, AUC 1.0 (training/train_ckd_india.py)."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "ckd_india_report_check"
    target = "ckd_india_report"
    input_schema = schema_for("ckd_india_report")
    high = "Possible kidney disease"
    low = "Kidney disease less likely"
    note = "An eGFR blood test and a urine albumin test confirm kidney disease."
