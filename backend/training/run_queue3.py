"""Cancer sets, detached from any session. From backend/. Waits for the oral and retinopathy queue first."""
import os
import subprocess
import sys
import time
from pathlib import Path

LOGS = Path(sys.argv[1])
while not (LOGS / "queue2_done.txt").exists():
    time.sleep(30)
K = "data/eval/kidneyct/x/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
JOBS = [("busi", "data/eval/busi/x/Dataset_BUSI_with_GT", "Breast ultrasound", "1", "_mask"),
        ("lungct", "data/eval/lungct/clean", "Lung CT", "1", ""),
        ("oralcancer", "data/eval/oralcancer/x/OralCancer", "Oral cancer from a photo", "1", ""),
        ("kidneyct", K, "Kidney CT", "1", "")]
for name, folder, title, near, skip in JOBS:
    if Path(f"workers/torch/models/multi/{name}.json").exists():
        continue
    with open(LOGS / f"fh_{name}.log", "a", encoding="utf-8") as f:
        subprocess.run([sys.executable, "training/train_folder_head.py", name, folder, title], stdout=f, stderr=subprocess.STDOUT, check=False,
                       env={**os.environ, "NEAR_DUPLICATES": near, "EXCLUDE_NAME": skip})
(LOGS / "queue3_done.txt").write_text("done")
