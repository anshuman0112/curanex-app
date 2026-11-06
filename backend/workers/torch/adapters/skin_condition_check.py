"""Skin condition shortlist from a photo, trained by the Curanex team on DermNet (training/train_folder_head.py).

Top-choice accuracy across 23 groups is 40%, so it never names one answer: it shows the three likeliest groups that passed
testing. Groups that failed (such as psoriasis and cellulitis) are never shown.
"""
import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from workers.common.adapter import Prediction
from workers.torch.multi_head import MultiHeadAdapter

SHORT = {
    "Hair Loss Photos Alopecia and other Hair Diseases": "Hair loss (alopecia)",
    "Nail Fungus and other Nail Disease": "Nail disease",
    "Acne and Rosacea Photos": "Acne or rosacea",
    "Atopic Dermatitis Photos": "Atopic dermatitis",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma or moles",
    "Poison Ivy Photos and other Contact Dermatitis": "Contact dermatitis",
    "Vasculitis Photos": "Vasculitis",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Actinic keratosis or skin cancer",
    "Seborrheic Keratoses and other Benign Tumors": "Benign skin growths",
    "Warts Molluscum and other Viral Infections": "Warts or viral skin infection",
    "Herpes HPV and other STDs Photos": "Herpes or HPV",
    "Vascular Tumors": "Vascular skin growths",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Ringworm or fungal infection",
    "Light Diseases and Disorders of Pigmentation": "Pigmentation disorders",
    "Eczema Photos": "Eczema",
    "Systemic Disease": "Skin signs of systemic disease",
    "Lupus and other Connective Tissue diseases": "Lupus or connective tissue disease",
    "Scabies Lyme Disease and other Infestations and Bites": "Scabies or bites",
    "Bullous Disease Photos": "Blistering disease"
}


class Adapter(MultiHeadAdapter):
    id = "skin_condition_check"
    name = "dermnet"
    grey = False
    wrong_image = "this looks like a scan or X-ray. Please upload a colour photo of the skin"

    def predict(self, inputs: dict) -> Prediction:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(inputs["image_b64"]))).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValueError("the image could not be read") from e
        if min(img.size) < self.min_side:
            raise ValueError("the image is too small; please upload a clearer one")
        rgb = np.asarray(img, dtype=np.float32)
        if float(np.abs(rgb[..., 0] - rgb[..., 2]).mean()) < self.min_colour:
            raise ValueError(self.wrong_image)
        p = self.session.run(None, {"image": self.prep(img)})[0][0]
        labels = self.meta["labels"]
        allowed = [i for i in range(len(labels)) if labels[i] in self.meta["passed"]]
        top = sorted(allowed, key=lambda i: -p[i])[:3]
        names = [SHORT[labels[i]] for i in top]
        warnings = ["This is a shortlist of possibilities, not an answer. Its first choice was right only about 4 times in 10.",
                    "Conditions such as psoriasis and cellulitis aren't covered."]
        scores = {n: round(float(p[i]), 3) for n, i in zip(names, top, strict=True)}
        return Prediction(model_id=self.id, label=f"Could be: {', '.join(names)}", warnings=warnings, probabilities=scores)
