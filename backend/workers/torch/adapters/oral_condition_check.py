"""Mouth and tooth condition from a photo, trained by the Curanex team on near-duplicate-free images (training/train_folder_head.py)."""
from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "oral_condition_check"
    name = "oral"
    found_label = "Looks most like {}"
    min_side = 150
    grey = False
    wrong_image = "this looks like an X-ray or scan. Please upload a colour photo of your mouth"
