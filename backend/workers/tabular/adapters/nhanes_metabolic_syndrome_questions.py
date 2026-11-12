"""Metabolic syndrome (ATP III, measured) from height, weight and simple questions. Verified on NHANES 2021-23 (sensitivity 0.918, AUC 0.802). Trained by the Curanex team on US CDC NHANES 2017-18."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_metabolic_syndrome_questions"
    target = "metabolic_syndrome_questions"
    input_schema = schema_for("metabolic_syndrome_questions")
    high = "Possible metabolic syndrome"
    low = "Metabolic syndrome less likely"
    note = "A waist measurement, blood pressure, lipid profile and HbA1c confirm it."
