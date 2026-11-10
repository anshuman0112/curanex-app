"""Fatty liver check from routine blood tests. Trained by the Curanex team (training/train_nhanes.py) on US CDC NHANES,
with liver fat measured by FibroScan."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_fatty_liver"
    target = "fatty_liver"
    input_schema = schema_for("fatty_liver")
    high = "Possible fatty liver"
    low = "Fatty liver less likely"
    note = "An ultrasound or FibroScan confirms fatty liver. Weight loss, exercise and avoiding alcohol help most people."
