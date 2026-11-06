"""Lung CT check, trained by the Curanex team (training/train_folder_head.py). Slices can share a patient, so scores may be optimistic."""
from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "lung_ct_check"
    name = "lungct"
    titles = {"Bengin": "Benign nodule", "Malignant": "Lung cancer", "Normal": "Normal"}
    clear = "Normal"
    found_label = "Possible {}"
    clear_label = "No clear signs of a lung nodule"
    grey = True
    dark_background = True
    dark_level, min_dark, min_edge_dark = 60, 0.3, 0.66  # these scans have a dark grey background, not black
    wrong_image = "this looks like a colour photo. Please upload the CT image"
