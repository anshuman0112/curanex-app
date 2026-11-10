"""Glioma grade from age and tumour gene tests (clinic-grade). Trained by the Curanex team (training/train_uci_clinic.py)
on UCI Glioma Grading (TCGA LGG and GBM)."""

from workers.tabular.clinic import ClinicAdapter, schema_for


class Adapter(ClinicAdapter):
    id = "clinic_glioma_grade"
    name = "glioma_grade"
    input_schema = schema_for("glioma_grade")
    high = "Profile fits glioblastoma (grade 4)"
    low = "Profile fits lower-grade glioma"
    note = "For neuro-oncology teams. Grading is decided by the pathologist (WHO criteria); this only reflects the gene profile."
