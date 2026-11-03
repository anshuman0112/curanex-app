# Plan for every model: how all 126 repos get used

| Group | Count | Plan | Where it's used |
|---|---|---|---|
| Wave 1 + 2 | 23 | Integrate now (`WORKFLOW.md` A3–A10) | AI Doctor in the app |
| Backlog | 34 | Download linked weights, or train (`TRAINING_PLAN.md`) | AI Doctor in the app |
| Hospital-only | 69 | **New routes for the input data** (this document) | App features + a clinic portal |

Training doesn't help the hospital-only repos. What they lack is **input**, not weights. So each one needs a way for the right data to reach it.

---

## Route A: Scan upload in the app (20 repos)

**Idea:** In India, patients usually get their MRI/CT on a CD or as a download link, and ultrasound reports with images. Let them upload those files.

| Kind | Repos | Feasible? |
|---|---|---|
| **2D classifiers** (one image is enough) | Brain tumor classification (`Seherin-Zaman`), osteoporosis (`BoranSekeroglu`), Alzheimer's (`aramis-lab`, check its input), Crohn's (`LNS98`) | ✅ First. Upload a DICOM file or a clear image of one slice |
| **3D segmentation** (needs the whole scan series, often several MRI sequences) | Stroke (`DeepIsles`, `ISLES22_SEALS`), MS (`maynord`, `uthmri`, `ivadomed`), brain tumor seg (`MONAI`), kidney/pancreas (`R-Super`, `PANORAMA`, `SlicerPolycysticKidneySeg`), kidney stone, liver fibrosis, prostate | ⚠️ Later. Needs a folder/zip of the DICOM series and a **GPU worker** with MONAI. Results take minutes, so they run as a background job with a notification |
| **Specialist ultrasound** (fetal echo, ovarian, CHD), DaT SPECT | `Pos4CHD`, `Cardium`, `czifan`, `parkinson-datscan` | ❌ for patients. Captured on clinic machines → Route E |

**What has to be built:**
- **App:** an "Upload a scan" screen that accepts `.dcm`, `.zip`, `.nii` or an image, with a preview using a simple DICOM viewer. The user picks the body part and scan type.
- **Backend:** DICOM handling with `pydicom` and `SimpleITK`. Strip patient-identifying DICOM tags on arrival (name, ID, birth date, hospital). Add a job queue (Redis + RQ) for slow models, a new **worker-medimg** (MONAI, GPU), and push notifications when a result is ready.
- **Safety:** results are framed as "for discussion with your radiologist" only. Never give an urgent "all clear".

## Route B: Wearables (3 repos)

| Repo | Data | How to get it |
|---|---|---|
| `EranZvuloni/MSD-generalization` (AF from ECG) | Single-lead ECG | Apple Watch ECG via **HealthKit**; other watches (Samsung, Fitbit, KardiaMobile) via their **exported ECG file** (PDF/CSV) uploaded in the app |
| `mazder1/ECG` (arrhythmia) | ECG | Same as above. Check its lead count. Many arrhythmia models expect 12-lead hospital ECGs, which would move it to Route E |
| `Subangkar/Simsig` (AF from PPG) | Raw PPG | Most watches don't share raw PPG. Option: **phone camera PPG** (finger over the flash for 30 s). Experimental; label it as such |

**What has to be built:** an Expo **development build** (HealthKit and Health Connect don't work in Expo Go), an ECG upload/import screen, resampling to the model's sample rate, and signal-quality checks (reject noisy recordings).

## Route C: Phone sensors (2 repos)

| Repo | Plan |
|---|---|
| `mohnabilfadl/Parkinson_Detection_Using_Voice_Analysis` | ✅ **Good fit.** The user says "aaah" for 5 seconds in a quiet room. The backend computes the voice features the model expects (jitter, shimmer, HNR and similar) with `praat-parselmouth`, then runs the model. **Caveat:** it was trained on studio-microphone recordings, and phone mics differ. Measure accuracy on phone recordings before switching it on, and show it as "screening, experimental" |
| `seantangth/cvpr2026-cv4chl` (cerebral palsy gait, children) | ❌ for now. Paediatric, needs clinical video protocols → Route E |

## Route D: Local health alerts (4 repos)

| Repo | Plan |
|---|---|
| `hunain6/Dengue-Risk-Prediction-System`, `horus-bot/Swasthya_Predective-system`, `BlackCatDisha/EpiSentinel` | ✅ Not a diagnosis. These predict **outbreak risk for an area** from weather and case data. Show a **"Health alert in your area"** card on Home ("Dengue risk is high in your district this week: remove standing water, use repellent"), using the user's district and a weather API. A user's location is never stored |
| `vaibhav-k/Dengue-Prediction-ML` (virus genomics) | ❌ Not usable from an app |

**What has to be built:** a nightly job that fetches weather (and public case counts where available), runs the models per district, and caches the result. A Home card in the app.

## Route E: Clinic portal for doctors and labs (34 repos)

**Idea:** these models belong in a clinic, not a phone. Microscope slides, endoscopy video, EEG files and ICU records are produced by clinics. Build **Curanex for Clinics**: a web dashboard, made from the same Expo web code or a separate React site, where verified doctors and lab staff log in and run models on the files they already have. A doctor is always in the loop, which is also the safest way to use these models.

| Data | Repos | Count |
|---|---|---|
| Blood smears / microscopy | Malaria ×2, leukemia ×2, sickle cell, IDA vs thalassemia | 6 |
| Histopathology / cytology / biopsy | LC25000, lung pathology, colorectal tissue, cervical cytology, breast biopsy features | 5 |
| Endoscopy / colonoscopy / laparoscopy | Polyps ×3, GI abnormalities ×2, Crohn's response, endometriosis ×2 | 8 |
| EEG / sleep signals | ADHD, OCD ×2, PTSD, depression/anxiety, sleep apnea | 6 |
| ICU / hospital records | Sepsis, AKI ×2, ICU mortality, readmission ×2, pressure injury ×2, hospital-acquired UTI | 9 |

**What has to be built:**
- Doctor accounts with verification (medical registration number), roles, and an audit log.
- Upload screens per data type:
  - Images: microscope photos or slide tiles.
  - Endoscopy: video frames.
  - EEG: `.edf` files, read with `mne`.
  - ICU/EHR: CSV upload with a column-mapping step.
- Batch runs, PDF reports, and a "doctor confirms / rejects" button on every result. Feedback is stored to measure real accuracy.
- Same workers and adapters as the app, plus `mne` for EEG and a GPU for endoscopy video.
- Optional later: the doctor shares a confirmed result into the patient's Curanex app.

## Route F: Foundation models (4 repos), parked

`CARDIAC-FM`, `OPENEye-FM`, `DB-EpiFM`, `FetalCLIP` are general-purpose models. Each needs **fine-tuning on a labelled dataset for a specific task** and a large GPU. The best later use: fine-tune `OPENEye-FM` to improve the eye models in the app. Park until there's GPU budget.

## Route G: Not feasible (2 repos)

`hears-lm` (genetic variants) and the PTSD methylation score need lab genetic data. Shown in the Model Library only.

---

## Order of work

| Phase | What | Models gained | Effort |
|---|---|---|---|
| 1 | Wave 1 + 2 integration | up to 23 | 1–2 weeks |
| 2 | Backlog Tracks 1–2 (download + tabular training) | ~15 | 1 week |
| 3 | Route C voice test + Route D local alerts | 4 | 3–5 days |
| 4 | Backlog Track 3 (image training) | ~12 | 1–2 weeks (GPU quota) |
| 5 | Route A scan upload (2D first) | 4, then up to 16 more with a GPU | 1 week + GPU work |
| 6 | Route E clinic portal | up to 34 | 3–4 weeks |
| 7 | Route B wearables (development build) | 2–3 | 1 week |

Total reachable: about **110 of 126**. The rest are parked (4) or not feasible (2), and a few will turn out to be broken or too weak once tested. **Every model still has to pass the same accuracy bar before it's switched on.**

---

## Prompts for Claude Code (one per route)

**Route C: voice screening**
```
Read docs/ALL_MODELS_PLAN.md Route C. Integrate mohnabilfadl/Parkinson_Detection_Using_Voice_Analysis:
find the exact voice features and their order from the repo, compute them from a 5-second sustained "aaah"
recording with praat-parselmouth in a new backend endpoint, and run the model in the right worker.
Add an app screen "Voice health check" with recording instructions (quiet room, 15 cm from the phone).
Label it "screening, experimental". Measure accuracy on any phone-recorded public set you can find,
or leave metrics null and verified=false. Stop and report.
```

**Route D: local alerts**
```
Read docs/ALL_MODELS_PLAN.md Route D. Integrate hunain6/Dengue-Risk-Prediction-System (and the other two
outbreak repos if compatible) as a nightly job: fetch weather per district from a weather API (key in .env),
compute risk, cache results. Add GET /alerts?district=. In the app, add a "Health alert in your area" card
on Home using the district from the profile (add a district field). Never store precise location. Stop and report.
```

**Route A: scan upload (2D first)**
```
Read docs/ALL_MODELS_PLAN.md Route A. Add scan upload: app screen accepting .dcm/.zip/.nii/images with
preview; backend /scans endpoint using pydicom/SimpleITK that strips identifying DICOM tags; select the
middle/representative slice for 2D models. Integrate Seherin-Zaman/Brain-Tumor-MRI-Dataset first
following WORKFLOW.md A3–A10. Results say "discuss with your radiologist". Plan (don't build) the
3D GPU worker and job queue in docs/SCAN_PIPELINE.md. Stop and report.
```

**Route E: clinic portal**
```
Read docs/ALL_MODELS_PLAN.md Route E. Design and scaffold "Curanex for Clinics": a web app (reuse the
design system), doctor sign-up with registration-number verification flag, roles, audit log, and upload
flows for (1) microscope images, (2) EEG .edf, (3) ICU CSV with column mapping. Integrate one model per
flow first: aabo5/Malaria-Detection, janaristevska/eeg-adhd-detection2, MikhailYadgarov/Sepsis-prediction,
following WORKFLOW.md A3–A10. Every result needs "doctor confirms / rejects", and the decision is stored.
Stop and report.
```

**Route B: wearables**
```
Read docs/ALL_MODELS_PLAN.md Route B. Set up an Expo development build. Add ECG import: Apple Health ECG via
HealthKit on iOS; PDF/CSV ECG export upload for other watches. Check the lead count each ECG repo expects.
Integrate EranZvuloni/MSD-generalization with resampling and a signal-quality check.
Label results "screening only". Stop and report.
```
