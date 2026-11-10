"""Gout risk from questions. Trained by the Curanex team (training/train_nhanes.py) on US CDC NHANES 2013-14 (doctor-
diagnosed gout), tested on 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_gout"
    target = "gout"
    input_schema = schema_for("gout")
    note = (
        "A uric acid blood test and a doctor's look at the painful joint confirm gout. A hot, red, very painful joint needs a doctor soon."
    )
