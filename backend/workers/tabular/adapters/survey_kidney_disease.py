"""Survey risk screener: Chronic kidney disease.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""

from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_kidney_disease"
    target = "kidney_disease"
    input_schema = schema_for("kidney_disease")
