"""Endoscopy finding adapters: one model, only the findings that passed an outside test are reported."""

import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from workers.common.adapter import Prediction
from workers.torch.multi_head import MultiHeadAdapter


class EndoFindingAdapter(MultiHeadAdapter):
    findings: dict = {}  # model class -> finding shown to the user
    clear_label = "No clear finding"
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
            if name in self.findings and p[i] >= self.meta["thresholds"][i]:
                found[self.findings[name]] = max(found.get(self.findings[name], 0), round(float(p[i]), 3))
        if found:
            top = max(found, key=found.get)
            return Prediction(model_id=self.id, label=f"Possible {top}", probabilities=found)
        clear = round(1 - float(max(p[labels.index(n)] for n in self.findings)), 3)
        return Prediction(model_id=self.id, label=self.clear_label, probabilities={self.clear_label: clear})
