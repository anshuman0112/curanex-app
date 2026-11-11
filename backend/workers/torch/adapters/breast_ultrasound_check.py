"""Breast ultrasound check, fully fine-tuned by the Curanex team on a Kaggle GPU (kaggle/make_breast_notebook.py) on BUS-BRA
(Brazil) and BUSI (Egypt), tested once on BrEaST (Poland): sensitivity 0.827, AUC 0.804 - Verified."""

from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "breast_ultrasound_check"
    name = "breast_us_ft"
    titles = {"benign": "Benign-looking lump", "malignant": "Lump that needs a biopsy"}
    clear = "benign"
    found_label = "Possible {}"
    clear_label = "Looks more like a benign lump"
    grey = None  # real scans often carry colour Doppler or coloured markers
