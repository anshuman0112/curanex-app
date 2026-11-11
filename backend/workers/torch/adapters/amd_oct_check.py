"""Age-related macular degeneration (AMD) from a retinal OCT scan. EfficientNet-B0 fully fine-tuned by the Curanex team on
Kermany 2018 OCT scans (CNV, DME, drusen, normal) on a Kaggle GPU. Only AMD is reported (wet AMD / CNV or drusen); the DME part
failed its outside test and is not shown. Verified on OCTDL (Russia) and confirmed on Duke / Srinivasan 2014 (USA)."""

import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from workers.common.adapter import Prediction
from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "amd_oct_check"
    name = "oct_ft"
    grey = True
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
        p = self.session.run(None, {"image": self.prep(img)})[0][0]
        amd = float(p[0] + p[2])  # CNV (wet AMD) + drusen
        t = self.meta["amd_threshold"]
        shown = amd / (2 * t) if amd < t else 1 - (1 - amd) / (2 * (1 - t))  # scaled so 50% is the flagging point
        label = "Possible age-related macular degeneration (AMD)" if amd >= t else "No clear signs of AMD"
        return Prediction(model_id=self.id, label=label, probabilities={"AMD signs": round(shown, 3), "No AMD signs": round(1 - shown, 3)})
