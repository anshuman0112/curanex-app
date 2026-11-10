"""Fetal distress from CTG values (clinic-grade). Trained by the Curanex team (training/train_uci_clinic.py) on UCI
Cardiotocography (Ayres de Campos et al.)."""

from workers.tabular.clinic import ClinicAdapter, schema_for


class Adapter(ClinicAdapter):
    id = "clinic_fetal_ctg"
    name = "fetal_ctg"
    input_schema = schema_for("fetal_ctg")
    high = "Suspicious or abnormal CTG pattern"
    low = "CTG pattern looks normal"
    note = "For use by a doctor or midwife reading the CTG. Reduced baby movements or bleeding in pregnancy need urgent care."
