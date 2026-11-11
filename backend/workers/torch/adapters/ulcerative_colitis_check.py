"""Ulcerative colitis on an endoscopy image, fully fine-tuned by the Curanex team on HyperKvasir on a Kaggle GPU
(kaggle/make_swap_notebooks.py). Verified on Kvasir v2: sensitivity 0.883, AUC 0.999."""

from workers.torch.endo_base import EndoFindingAdapter


class Adapter(EndoFindingAdapter):
    id = "ulcerative_colitis_check"
    name = "endoscopy_hk"
    findings = {"ulcerative-colitis": "ulcerative colitis"}
    clear_label = "No clear signs of ulcerative colitis"
