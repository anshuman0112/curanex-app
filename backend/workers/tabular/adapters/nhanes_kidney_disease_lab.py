"""Hidden kidney disease risk (eGFR < 60 or urine albumin/creatinine >= 30) from questions only. Trained by the Curanex
team (training/train_nhanes.py) on US CDC NHANES 2017-18, tested on 2021-23."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_kidney_disease_lab"
    target = "kidney_disease_lab"
    input_schema = schema_for("kidney_disease_lab")
    note = "A creatinine (eGFR) blood test and a urine albumin test are cheap and show kidney damage early."
