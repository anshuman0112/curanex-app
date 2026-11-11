"""Kidney CT check, trained by the Curanex team (training/train_folder_head.py). Slices can share a patient, so scores may be optimistic."""

from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "kidney_ct_check"
    name = "kidneyct"
    titles = {"Cyst": "Kidney cyst", "Normal": "Normal", "Stone": "Kidney stone", "Tumor": "Kidney tumor"}
    clear = "Normal"
    found_label = "Possible {}"
    clear_label = "No clear kidney finding"
    grey = True
    dark_background = True
    wrong_image = "this looks like a colour photo. Please upload the CT image"
