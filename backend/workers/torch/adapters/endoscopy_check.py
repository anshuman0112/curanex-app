"""Endoscopy image check, trained by the Curanex team on Kvasir (training/train_folder_head.py).
Polyps and ulcerative colitis passed testing. Esophagitis missed the bar, so it's shown as a possibility and not counted as covered.
"""
import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from workers.common.adapter import Prediction
from workers.torch.multi_head import MultiHeadAdapter

FINDINGS = {"polyps": "polyp", "dyed-lifted-polyps": "polyp", "ulcerative-colitis": "ulcerative colitis", "esophagitis": "esophagitis"}


class Adapter(MultiHeadAdapter):
    id = "endoscopy_check"
    name = "kvasir"
    grey = False
    wrong_image = "this looks like a grey scan. Please upload a colour endoscopy image"

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
        found = {}
        for i, name in enumerate(labels):
            if name in FINDINGS and p[i] >= self.meta["thresholds"][i]:
                found[FINDINGS[name]] = max(found.get(FINDINGS[name], 0), round(float(p[i]), 3))
        if found:
            top = max(found, key=found.get)
            return Prediction(model_id=self.id, label=f"Possible {top}", probabilities=found)
        clear = round(1 - float(max(p[labels.index(n)] for n in FINDINGS)), 3)
        return Prediction(model_id=self.id, label="No clear finding", probabilities={"No clear finding": clear})
