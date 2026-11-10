"""Survey risk screener: Memory and concentration problems.

Trained by the Curanex team (training/train_brfss.py) on the US CDC BRFSS survey.
"""
from workers.tabular.survey import SurveyAdapter, schema_for


class Adapter(SurveyAdapter):
    id = "survey_memory_problems"
    target = "memory_problems"
    input_schema = schema_for("memory_problems")
