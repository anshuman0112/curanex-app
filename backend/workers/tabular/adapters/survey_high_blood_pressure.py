"""Survey risk screener: High blood pressure.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""
from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_high_blood_pressure"
    target = "high_blood_pressure"
    input_schema = schema_for("high_blood_pressure")
