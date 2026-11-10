"""Mammogram mass check from the radiologist's findings (clinic-grade). Trained by the Curanex team
(training/train_uci_clinic.py) on UCI Mammographic Mass (Elter et al.)."""

from workers.tabular.clinic import ClinicAdapter, schema_for


class Adapter(ClinicAdapter):
    id = "clinic_mammo_mass"
    name = "mammo_mass"
    input_schema = schema_for("mammo_mass")
    high = "Findings fit a malignant mass"
    low = "Findings fit a benign mass"
    note = "For use by a radiologist or doctor. A biopsy decides; this is a second look at the reported findings only."
