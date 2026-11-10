"""Liver cirrhosis in chronic liver disease (clinic-grade). Trained by the Curanex team (training/train_uci_clinic.py) on
UCI Cirrhosis Patient Survival (Mayo Clinic PBC trial)."""

from workers.tabular.clinic import ClinicAdapter, schema_for


class Adapter(ClinicAdapter):
    id = "clinic_cirrhosis"
    name = "cirrhosis"
    input_schema = schema_for("cirrhosis")
    high = "Possible cirrhosis"
    low = "Cirrhosis less likely"
    note = "For doctors treating chronic liver disease. Vomiting blood, black stools or confusion need emergency care."
