"""Osteoporosis risk (hip DXA T-score -2.5 or lower) from age, sex, weight, height and smoking. Trained by the Curanex team
(training/train_nhanes.py) on US CDC NHANES 2013-14, tested on 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_osteoporosis"
    target = "osteoporosis"
    input_schema = schema_for("osteoporosis")
    note = "A DXA bone density scan is the real test. Calcium, vitamin D, weight-bearing exercise and not smoking protect bones."
