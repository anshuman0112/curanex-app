"""Breast ultrasound check, trained by the Curanex team (training/train_breast2.py) on BUS-BRA (Brazil) and BUSI (Egypt),
tested once on BrEaST (Poland). Unverified: outside sensitivity 0.898, AUC 0.769."""
from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "breast_ultrasound_check"
    name = "breast_us"
    titles = {"benign": "Benign-looking lump", "malignant": "Lump that needs a biopsy"}
    clear = "benign"
    found_label = "Possible {}"
    clear_label = "Looks more like a benign lump"
    grey = None  # real scans often carry colour Doppler or coloured markers (42 of 252 in the Polish test set)
