"""Oesophagitis on an endoscopy image of the oesophagus-stomach junction, trained by the Curanex team on Kvasir
(training/train_folder_head.py). Unverified: tested on held-out Kvasir images only."""

from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "esophagitis_check"
    name = "esophagitis"
    titles = {"Esophagitis": "Oesophagitis", "Normal_z_line": "Normal oesophagus junction"}
    clear = "Normal_z_line"
    found_label = "Possible {}"
    clear_label = "No clear signs of oesophagitis"
    grey = False
    wrong_image = "this looks like a grey image. Please upload a colour endoscopy image"
