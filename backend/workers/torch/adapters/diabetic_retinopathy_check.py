"""Diabetic retinopathy grade from a retina photo, trained by the Curanex team on APTOS (training/train_folder_head.py)."""
from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "diabetic_retinopathy_check"
    name = "dr"
    titles = {"Mild": "Mild diabetic retinopathy", "Moderate": "Moderate diabetic retinopathy", "No_DR": "No diabetic retinopathy",
              "Proliferate_DR": "Proliferative diabetic retinopathy", "Severe": "Severe diabetic retinopathy"}
    clear = "No_DR"
    found_label = "Possible {}"
    clear_label = "No clear signs of diabetic retinopathy"
    grey = False
    min_colour = 3.0
    wrong_image = "this looks like a grey scan. Please upload a colour retina photo from an eye clinic"
