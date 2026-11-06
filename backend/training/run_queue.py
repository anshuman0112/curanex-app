"""Runs the remaining trainings one after another, detached from any session, and skips finished ones. From backend/."""
import subprocess
import sys
from pathlib import Path

LOGS = Path(sys.argv[1])
HEADS = [("malaria", "data/eval/malaria/x/cell_images/cell_images", "Malaria blood smear cell"),
         ("dermnet", "data/eval/dermnet/x", "Skin disease from a clinical photo"),
         ("kvasir", "data/eval/kvasir/x/kvasir-dataset", "Digestive tract endoscopy finding")]
TINY = ["organamnist", "octmnist", "pathmnist"]
for name, folder, title in HEADS:
    if not Path(f"workers/torch/models/multi/{name}.json").exists():
        with open(LOGS / f"fh_{name}.log", "a", encoding="utf-8") as f:
            subprocess.run([sys.executable, "training/train_folder_head.py", name, folder, title], stdout=f, stderr=subprocess.STDOUT, check=False)
for name in TINY:
    if not Path(f"workers/torch/models/medmnist/{name}.json").exists():
        with open(LOGS / f"mm_{name}.log", "a", encoding="utf-8") as f:
            subprocess.run([sys.executable, "training/train_medmnist.py", name], stdout=f, stderr=subprocess.STDOUT, check=False)
(LOGS / "queue_done.txt").write_text("done")
