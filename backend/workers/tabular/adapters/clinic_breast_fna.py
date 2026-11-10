"""Breast lump fine-needle aspirate (FNA) cytology scores (clinic-grade). Trained by the Curanex team
(training/train_uci_clinic.py) on UCI Breast Cancer Wisconsin (Original)."""

from workers.tabular.clinic import ClinicAdapter, schema_for


class Adapter(ClinicAdapter):
    id = "clinic_breast_fna"
    name = "breast_fna"
    input_schema = schema_for("breast_fna")
    high = "Cytology suggests malignancy"
    low = "Cytology looks benign"
    note = "For use by a pathologist or doctor. The pathologist's report and a biopsy decide; this is a second look only."
