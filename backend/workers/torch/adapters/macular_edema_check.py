"""Diabetic macular oedema from a retina photo, fully fine-tuned by the Curanex team on Messidor-2 (France) on a Kaggle GPU.
Verified on all 455 graded IDRiD photos (India): sensitivity 0.875, AUC 0.943."""

from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "macular_edema_check"
    name = "dme_messidor"
    titles = {"Macular_edema": "Diabetic macular oedema", "No_macular_edema": "No macular oedema"}
    clear = "No_macular_edema"
    found_label = "Possible {}"
    clear_label = "No clear signs of macular oedema"
    grey = False
    min_colour = 3.0
    wrong_image = "this looks like a grey scan. Please upload a colour retina photo from an eye clinic"
