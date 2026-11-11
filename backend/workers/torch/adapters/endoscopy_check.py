"""Colon polyp check on an endoscopy image, trained by the Curanex team on Kvasir (training/train_folder_head.py).
Verified for polyps on HyperKvasir (sensitivity 0.876, AUC 0.98). Its other findings are not reported: ulcerative colitis and
oesophagitis have their own checks, with models that passed their own outside tests."""

from workers.torch.endo_base import EndoFindingAdapter


class Adapter(EndoFindingAdapter):
    id = "endoscopy_check"
    name = "kvasir"
    findings = {"polyps": "colon polyp", "dyed-lifted-polyps": "colon polyp"}
    clear_label = "No clear sign of a polyp"
