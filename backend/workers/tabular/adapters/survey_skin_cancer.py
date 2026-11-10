"""Survey risk screener: Skin cancer (not melanoma).

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""
from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_skin_cancer"
    target = "skin_cancer"
    input_schema = schema_for("skin_cancer")
