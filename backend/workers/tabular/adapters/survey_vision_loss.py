"""Survey risk screener: Serious vision problems.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""
from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_vision_loss"
    target = "vision_loss"
    input_schema = schema_for("vision_loss")
