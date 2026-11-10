"""Survey risk screener: Serious hearing difficulty.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""
from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_hearing_difficulty"
    target = "hearing_difficulty"
    input_schema = schema_for("hearing_difficulty")
