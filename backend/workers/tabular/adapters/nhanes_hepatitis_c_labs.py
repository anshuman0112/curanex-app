"""Hepatitis C infection (HCV RNA positive) from a liver function test and platelets. Verified on NHANES 2021-23 (sensitivity 0.889, AUC 0.890; 27 cases). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_hepatitis_c_labs"
    target = "hepatitis_c_labs"
    input_schema = schema_for("hepatitis_c_labs")
    high = "Possible hepatitis C"
    low = "Hepatitis C less likely"
    note = "An anti-HCV antibody test, then an HCV RNA test, confirms it."
