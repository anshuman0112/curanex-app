"""Low muscle mass (DXA appendicular lean mass / BMI below 0.789 men / 0.512 women, FNIH) in adults aged 20-59, from simple
questions. Trained by the Curanex team on US CDC NHANES 2011-14 (training/train_nhanes_p.py); Verified on NHANES 2017-18
(sensitivity 0.819, AUC 0.942)."""

from workers.tabular.nhanes import NhanesAdapter, schema_for


class Adapter(NhanesAdapter):
    id = "nhanes_low_muscle_mass"
    target = "low_muscle_mass"
    input_schema = schema_for("low_muscle_mass")
    high = "Possible low muscle mass"
    low = "Low muscle mass less likely"
    note = "A body composition scan (DXA) or a clinic muscle check confirms it."
