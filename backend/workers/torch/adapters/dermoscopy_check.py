"""Dermoscopy skin lesion check (clinic-grade), trained by the Curanex team on HAM10000 (training/train_dermoscopy.py).
Unverified: tested on held-out HAM10000 lesions only."""

from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "dermoscopy_check"
    name = "dermoscopy"
    titles = {
        "not_suspicious": "Lesion that looks harmless",
        "suspicious": "Suspicious lesion (melanoma, basal cell carcinoma or actinic keratosis)",
    }
    clear = "not_suspicious"
    found_label = "Possible {}"
    clear_label = "Lesion doesn't look suspicious on this image"
    grey = False
    wrong_image = "this looks like a grey image. Please upload a colour dermoscope image of the skin spot"
