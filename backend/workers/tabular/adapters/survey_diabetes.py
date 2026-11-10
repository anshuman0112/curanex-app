"""Survey risk screener: Type 2 diabetes.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""

from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_diabetes"
    target = "diabetes"
    input_schema = schema_for("diabetes")
