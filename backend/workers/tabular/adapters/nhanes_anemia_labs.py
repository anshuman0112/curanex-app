"""Anaemia (low haemoglobin) from MCV, MCH, RDW and platelets, without haemoglobin. Verified on NHANES 2021-23 (sensitivity 0.844, AUC 0.814). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_anemia_labs"
    target = "anemia_labs"
    input_schema = schema_for("anemia_labs")
    high = "Possible anaemia"
    low = "Anaemia less likely"
    note = "A haemoglobin test confirms anaemia."
