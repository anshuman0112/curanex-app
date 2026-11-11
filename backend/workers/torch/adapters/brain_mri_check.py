"""Brain MRI tumor check, trained by the Curanex team (training/train_folder_head.py)."""

import numpy as np

from workers.torch.multi_head import MultiHeadAdapter

MIN_DARK = 0.2  # MRI slices sit on a black background; chest and knee X-rays don't
MIN_EDGE_DARK = 0.45


class Adapter(MultiHeadAdapter):
    id = "brain_mri_check"
    name = "brain"
    titles = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary tumor"}
    clear = "notumor"
    found_label = "Possible brain tumor, looks most like {}"
    clear_label = "No clear tumor signs"
    grey = True
    wrong_image = "this looks like a colour photo. Please upload the MRI image"

    def check(self, rgb: np.ndarray) -> None:
        g = rgb.mean(axis=2)
        b = max(1, int(min(g.shape) * 0.08))
        edge = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
        if float((g < 20).mean()) < MIN_DARK and float((edge < 20).mean()) < MIN_EDGE_DARK:
            raise ValueError("this doesn't look like a brain MRI slice. Please upload the MRI image")
