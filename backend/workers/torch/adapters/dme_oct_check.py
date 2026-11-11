"""Diabetic macular oedema (DME) from a retinal OCT scan, with the same fine-tuned Kermany model as the AMD check. The DME
threshold was set once on OCTDL and Verified once on Duke / Srinivasan 2014 (eye level sensitivity 1.0, AUC 1.0; scan level
sensitivity 0.826, AUC 0.971)."""

import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from workers.common.adapter import Prediction
from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "dme_oct_check"
    name = "oct_ft"
    wrong_image = "this looks like a colour photo. Please upload a retinal OCT scan (the grey cross-section image from an eye clinic)"

    def predict(self, inputs: dict) -> Prediction:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(inputs["image_b64"]))).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValueError("the image could not be read") from e
        if min(img.size) < self.min_side:
            raise ValueError("the image is too small; please upload a clearer one")
        rgb = np.asarray(img, dtype=np.float32)
        if float(np.abs(rgb[..., 0] - rgb[..., 2]).mean()) > 12:
            raise ValueError(self.wrong_image)
        p = float(self.session.run(None, {"image": self.prep(img)})[0][0][1])  # DME
        t = self.meta["dme_threshold"]
        shown = p / (2 * t) if p < t else 1 - (1 - p) / (2 * (1 - t))
        label = "Possible diabetic macular oedema (DME)" if p >= t else "No clear signs of DME"
        return Prediction(model_id=self.id, label=label, probabilities={"DME signs": round(shown, 3), "No DME signs": round(1 - shown, 3)})
