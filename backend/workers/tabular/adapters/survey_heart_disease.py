"""Survey risk screener: Coronary heart disease or heart attack.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""

from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_heart_disease"
    target = "heart_disease"
    input_schema = schema_for("heart_disease")
