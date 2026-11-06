"""Oral disease, then diabetic retinopathy, detached from any session. From backend/."""
import os
import subprocess
import sys
from pathlib import Path

LOGS = Path(sys.argv[1])
JOBS = [("oral", "data/eval/oral/clean", "Mouth and tooth condition from a photo", "1"), ("dr", "data/eval/dr/colored_images", "Diabetic retinopathy grade from a retina photo", "0")]
for name, folder, title, near in JOBS:
    if Path(f"workers/torch/models/multi/{name}.json").exists():
        continue
    with open(LOGS / f"fh_{name}.log", "a", encoding="utf-8") as f:
        subprocess.run([sys.executable, "training/train_folder_head.py", name, folder, title], stdout=f, stderr=subprocess.STDOUT, check=False, env={**os.environ, "NEAR_DUPLICATES": near})
(LOGS / "queue2_done.txt").write_text("done")
