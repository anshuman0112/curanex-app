"""Low bone density (hip DXA T-score -1 or lower), age 50+. Trained by the Curanex team (training/train_nhanes.py) on
US CDC NHANES 2013-14, tested on 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_low_bone_density"
    target = "low_bone_density"
    input_schema = schema_for("low_bone_density")
    note = "A DXA bone density scan is the real test."
