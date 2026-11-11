"""Malaria check on one cropped blood-smear cell, fully fine-tuned by the Curanex team on a Kaggle GPU (kaggle/make_malaria_notebook.py).
Verified on BBBC041 (another lab, stain and parasite species): sensitivity 0.877, AUC 0.925."""

from workers.torch.multi_head import MultiHeadAdapter


class Adapter(MultiHeadAdapter):
    id = "malaria_cell_check"
    name = "malaria_ft"
    titles = {"Parasitized": "Malaria parasite", "Uninfected": "No parasite"}
    clear = "Uninfected"
    found_label = "{} signs found in this cell"
    clear_label = "No parasite signs in this cell"
    min_side = 40
    grey = False
    min_colour = 1.0  # stained cells can be faintly coloured, and only a fully grey image is refused
    wrong_image = "this looks like a grey scan. Please upload a colour image of a stained blood cell"
