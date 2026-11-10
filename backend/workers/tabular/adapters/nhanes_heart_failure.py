"""Heart failure risk from questions. Trained by the Curanex team (training/train_nhanes.py) on US CDC NHANES 2017-18
(doctor-diagnosed congestive heart failure), tested on 2021-23."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_heart_failure"
    target = "heart_failure"
    input_schema = schema_for("heart_failure")
    note = "Severe breathlessness, chest pain or fainting is an emergency: call 112. An echocardiogram confirms heart failure."
