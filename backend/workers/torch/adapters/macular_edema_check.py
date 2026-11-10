"""Diabetic macular oedema from a retina photo, trained by the Curanex team on IDRiD (training/train_folder_head.py).
Unverified: tested on IDRiD's own held-out test images only."""

from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "macular_edema_check"
    name = "idrid_dme"
    titles = {"Macular_edema": "Diabetic macular oedema", "No_macular_edema": "No macular oedema"}
    clear = "No_macular_edema"
    found_label = "Possible {}"
    clear_label = "No clear signs of macular oedema"
    grey = False
    min_colour = 3.0
    wrong_image = "this looks like a grey scan. Please upload a colour retina photo from an eye clinic"
