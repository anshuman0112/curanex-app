"""Survey risk screener: Arthritis.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""

from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_arthritis"
    target = "arthritis"
    input_schema = schema_for("arthritis")
