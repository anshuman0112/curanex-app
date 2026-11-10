# Scan pipeline (Route A)

## Built (2D, 2026-11-10)

- `POST /upload` accepts a DICOM file or a `.zip` of one series as well as photos (`backend/api/scans.py`).
- A series is sorted by slice position and the **middle slice** is used, which is what the 2D models expect.
- The pixels are rendered with the display window stored in the file (rescale slope/intercept, window centre/width, MONOCHROME1 inverted), or the 1st–99th percentile range when there is none.
- **Only the pixels are kept.** The image is re-rendered and saved as a new JPEG, so the DICOM header (name, ID, birth date, hospital and every other tag) never reaches storage. A file marked `BurnedInAnnotation = YES` is refused, because printed text would survive.
- Limits: 60 MB upload, 1000 files and 300 MB unzipped per `.zip`. Compressed transfer syntaxes the server can't decode get a message asking for an uncompressed export.
- In the app, CT, brain MRI and X-ray upload cards have a "Scan file (DICOM)" button.
- The scan goes to the existing live models (kidney CT, lung CT, brain MRI, chest X-ray, knee, arm/hand). Their image guards still apply, so a small or wrong-type slice is refused.
- Check: real kidney-CT and brain-MRI test images wrapped in DICOM gave exactly the same labels and scores as the original images (stone 0.991, normal 1.0, meningioma 0.997). pydicom's 64–128 px sample files were correctly refused as too small.

Not built: `.nii` (NIfTI) upload, a DICOM viewer in the app, and choosing a slice other than the middle one.

## Planned, not built: 3D models on a GPU worker

The 3D repos in ALL_MODELS_PLAN Route A (stroke, MS, tumour segmentation, kidney/pancreas, prostate) need the whole series, often several MRI sequences, and minutes of GPU time.

1. **Upload:** the same `/upload` path stores the anonymised series as a NIfTI volume (SimpleITK), not one slice. Identifying tags are dropped the same way, and the volume is kept only until the job ends.
2. **Queue:** `POST /scans/jobs` puts a job on Redis (RQ). It returns a job id at once, and the app polls `GET /scans/jobs/{id}` (or receives a push notification on a development build).
3. **worker-medimg:** a new Docker worker with MONAI and CUDA. It has the same `/health`, `/models` and `/predict` contract as the other workers, loads one model at a time and has a 10-minute timeout.
4. **Results:** a segmentation overlay on three slices plus volumes (for example lesion volume in ml), always worded "discuss with your radiologist", never as a diagnosis.
5. **Switch-on bar:** as for every model, sensitivity ≥ 0.80 and AUC ≥ 0.80, or a Dice score agreed in advance, on a dataset from a different hospital.
6. **Hardware:** this laptop has no GPU, so this needs a cloud GPU (for example a free Kaggle or Colab session for testing, and a paid GPU host for a demo).
