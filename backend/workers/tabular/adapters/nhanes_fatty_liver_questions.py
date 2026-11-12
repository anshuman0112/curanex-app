"""Fatty liver (FibroScan CAP >= 285) from waist, BMI and simple questions, no blood tests. Verified on NHANES 2021-23 (sensitivity 0.866, AUC 0.827). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_fatty_liver_questions"
    target = "fatty_liver_questions"
    input_schema = schema_for("fatty_liver_questions")
    high = "Possible fatty liver"
    low = "Fatty liver less likely"
    note = "An ultrasound or FibroScan confirms fatty liver."
