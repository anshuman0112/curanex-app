"""Hidden (undiagnosed) diabetes from a routine blood test, without sugar tests. Trained by the Curanex team on US CDC NHANES
2017-18 (HbA1c >= 6.5% in people never told they have diabetes); Verified on NHANES 2021-23 (sensitivity 0.954, AUC 0.815)."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_hidden_diabetes_labs"
    target = "hidden_diabetes_labs"
    input_schema = schema_for("hidden_diabetes_labs")
    high = "Possible hidden diabetes"
    low = "Hidden diabetes less likely"
    note = "An HbA1c or fasting blood sugar test confirms diabetes. This check does not replace it."
