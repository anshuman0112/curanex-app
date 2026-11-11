"""Oesophagitis on an endoscopy image, fully fine-tuned by the Curanex team on HyperKvasir on a Kaggle GPU
(kaggle/make_swap_notebooks.py). Verified on Kvasir v2: sensitivity 0.801, AUC 0.931."""

from workers.torch.endo_base import EndoFindingAdapter


class Adapter(EndoFindingAdapter):
    id = "esophagitis_check"
    name = "endoscopy_hk"
    findings = {"esophagitis": "oesophagitis"}
    clear_label = "No clear signs of oesophagitis"
