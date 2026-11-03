# Training plan: getting working models for the 34 backlog repos

The backlog repos have the code, but no usable trained weights in the repo. There are four ways to get a working model, depending on the repo. **Always try Track 1 first.** It needs no training.

> Hospital-only repos (69) are a different problem. They need MRI, CT, EEG or ICU **input data** that a phone user can't give. Training doesn't change that, so they're not covered here.

---

## Track 1: download the weights (no training)

Several READMEs link to the weights on another site. The audit found these:

| Repo | Task | Where the weights are |
|---|---|---|
| `sibisai/skin-scan` | Skin conditions (phone photo) | Hugging Face |
| `deettoh/skin-lesion-classification` | Melanoma / skin lesions | Hugging Face |
| `sfu-mial/skin-IAV` | Skin lesions | Hugging Face / Zenodo |
| `Whitedevil71/FundusNext` | Glaucoma (fundus) | Hugging Face |
| `li-xirong/mmc-amd` | Macular degeneration | Google Drive |
| `darwinai/TuberculosisNet` | Tuberculosis (X-ray) | Link in README (check) |
| `nyukat/breast_cancer_classifier`, `nyukat/gmic` | Mammography | Download script in the repo |

**Steps:** read the README → download the weights into `backend/vendor/<id>/weights/` → record the source URL and SHA-256 → continue from integration step A4 (sandbox load).

Since you're in touch with the authors, you can also **ask any backlog author for their trained file**. That's the fastest route of all.

---

## Track 2: train tabular models on your laptop (minutes each)

These use form-type data (lab values, symptoms), so a normal laptop is enough and no GPU is needed.

| Repo | Task | Data needed |
|---|---|---|
| `vineet416/Chronic-Kidney-Disease-Prediction` | Chronic kidney disease | Blood and urine values (UCI CKD dataset is the usual one) |
| `yehonatanke/Thyroid-Predict-ML` | Hyper/hypothyroidism | Thyroid labs (TSH, T3, T4) |
| `SakH1075/...Thalassemia...CBC` | Thalassemia | CBC values |
| `djparente/uti-ml` | Urinary tract infection | Urinalysis + symptoms |
| `rick0110/cervical_cancer` | Cervical cancer risk | Risk-factor questionnaire |
| `PRAISE-Lab-Repository/PEPrML` | Severe preeclampsia | Maternal clinical values |
| `lanagarmire/PE_delivery` | Preeclampsia delivery timing | Maternal clinical values |
| `Vankudoth-Sunil-Kumar/DiseasePrediction` | Liver disease (+ others) | Liver function tests |
| `uxlabspk/NeuroVision` | Autism trait screening | Questionnaire (has a `.tflite` file: try Track 1 first) |
| `ifyoungnet/MedGDM` | Gestational diabetes | Has a KNIME model file: retrain in Python instead |
| `resplab/accept` | COPD exacerbation | Written in R: port the published model equations to Python, or skip |

**Where the data comes from:** each repo's README or notebook names its dataset. Claude Code reads it, finds the public download (UCI, Kaggle, PhysioNet, the paper's supplement) and records the dataset licence. If the data needs an approved account (PhysioNet credentialing, for example), it stops and tells you.

---

## Track 3: fine-tune image models on a free cloud GPU (1–3 hours each)

Image models need a GPU. Free options:
- **Kaggle Notebooks**: a free GPU with a weekly hour quota, and many medical image datasets are already hosted there.
- **Google Colab**: a free T4 GPU, with session limits.

Use **transfer learning**: start from a model already trained on millions of everyday images (EfficientNet, ResNet, or a ViT from `timm`), then fine-tune it on the medical dataset. That's far faster and needs far less data than training from scratch.

| Repo | Task | Typical public data |
|---|---|---|
| `smon2006/Skin-Disease-Detection` | Common skin diseases (phone photo) | Dataset named in the repo (Kaggle) |
| `nawaz127/A-Multilayer-...Skin-Disease-` | Multiclass skin disease | Named in the repo |
| `Duggineniakhil/oral-cancer-detection` | Oral cancer (phone photo) | Named in the repo |
| `MarcoParola/oral_segmentation` | Oral lesions | Drive/Kaggle link in the README |
| `ShaonINT/Glaucoma_Detection` | Glaucoma | Kaggle (named in the notebook) |
| `anchor-neuro/retguard` | Retinal screening | Named in the repo |
| `CASALab-Unisa/Dino2-DR` | Diabetic retinopathy | Public DR grading sets (APTOS / EyePACS style) |
| `pradamehavannan/OCT-Based-Retinal-...` | OCT retinal disease | Kaggle OCT dataset (named in the repo) |
| `sathvika-09/TB-detection` | Tuberculosis X-ray | Public TB chest X-ray sets |
| `Holy-Mahdi/ibids-xray-fracture-detection` | Fractures (YOLO) | Named in the repo |
| `PingjunChen/GradingKneeOA` | Knee OA grade | Public knee X-ray (KL grade) sets |
| `m-aljasem/pediatric-radiology-ai` | Bone age | Public hand X-ray bone-age set |

**How it works:** Claude Code writes one Kaggle/Colab notebook per model (`training/notebooks/<id>.ipynb`). You upload it, attach the dataset, press **Run all**, and download the result: `model.onnx` plus `metrics.json`. Claude Code then integrates it from step A6.

---

## Track 4: defer for now

| Repo | Why |
|---|---|
| `Eeman1113/dental_study` | MedGemma LoRA: needs a large GPU to run, not just to train |
| `Bhuvanaa28/Sentiment-Analysis-for-Depression-Detection` | Mental-health text. A wrong "no risk" result is dangerous. If you add it, it must only ever route to support and crisis resources, never give a verdict |
| `MarcoParola/detection_framework` | Research framework, not a single model |

---

## The standard training recipe (every Track 2 and 3 model)

Claude Code writes `backend/training/` with one shared recipe so every model is trained and measured the same way.

1. **Get the data:** download it, record the source URL, licence, row/image count and class balance. Never commit patient data to Git.
2. **Clean it:** follow the repo's own preprocessing (read from its notebook). Fix obvious problems like duplicate rows or impossible values.
3. **Split it before anything else:** 70% train / 15% validation / 15% test, stratified by class. If the same patient appears more than once, keep all their rows in one split. **The test set is touched only once, at the end.**
4. **Train:**
   - Tabular: logistic regression as a baseline, then XGBoost or LightGBM, with 5-fold cross-validation on the training set.
   - Images: a pretrained EfficientNet-B0 or ResNet-50, 224 px input, augmentation (flips, small rotations, brightness), class weights if the classes are unbalanced, early stopping on the validation score.
5. **Calibrate:** make sure "70%" really means about 70% (Platt or isotonic calibration). This matters because the app shows percentages to users.
6. **Measure on the test set:** accuracy, sensitivity, specificity, AUC, and a confusion matrix. Report per-class numbers for multi-class models.
7. **Explain:** SHAP for tabular models, Grad-CAM for images. Check that the model looks at sensible things (the lesion, not a ruler or text in the corner).
8. **Export:** `model.onnx` + `preprocess.json` + `metrics.json` + `model_card.md` (data, method, results, limitations, credit to the original repo author).
9. **Integrate:** continue from WORKFLOW.md step A6 (adapter → registry → model search test → switch on).

**Accuracy bar before switching a model on:** sensitivity ≥ 0.80 and AUC ≥ 0.80 on the held-out test set. Anything lower stays "Unverified" in the app, or isn't switched on at all. Never report the validation or training score as the result.

**Honest labelling:** these models are *"trained by the Curanex team using the method from <repo>, on <dataset>"*. The Model Library and result card say exactly that.

---

## Time estimate

| Track | Per model | All of it |
|---|---|---|
| 1 Download | 15–45 min | ~7 models, about half a day |
| 2 Tabular | 30–90 min | ~10 models, 1–2 days |
| 3 Image | 2–4 h (mostly waiting for the GPU) | ~12 models, 3–5 days, split around free GPU quotas |

---

## Prompts for Claude Code

**T0: sort the backlog** (after the clone-all prompt has run)
```
Read docs/TRAINING_PLAN.md and docs/INTEGRATION_REPORT.md. For every backlog repo in backend/vendor/,
read its README, notebooks and scripts and decide its track: 1 Download (weights linked somewhere), 2 Tabular,
3 Image, or 4 Defer. For tracks 2 and 3, find the exact dataset the repo used: name, public URL, licence,
size, and whether it needs an account. Write docs/TRAINING_QUEUE.md as a table sorted by track, then by
expected value to users. Don't download any datasets yet. Stop and report.
```

**T1: download linked weights** (repeat per repo)
```
For <REPO> (Track 1 in docs/TRAINING_QUEUE.md): download the weights from the link in its README into
backend/vendor/<id>/weights/, record the source URL and SHA-256 in the registry audit field, then integrate it
following docs/WORKFLOW.md steps A3 to A10. Apply the 45-minute rule. Stop and report.
```

**T2: build the training recipe** (once)
```
Create backend/training/ implementing the standard recipe in docs/TRAINING_PLAN.md:
common/data.py (download, record source and licence, stratified/grouped split saved to disk),
common/tabular.py (baseline + XGBoost/LightGBM, 5-fold CV, calibration, SHAP),
common/evaluate.py (test-set metrics, confusion matrix, metrics.json),
common/export.py (ONNX + preprocess.json + model_card.md),
and a notebook template training/notebooks/_image_template.ipynb for Kaggle/Colab
(timm EfficientNet-B0 transfer learning, augmentation, early stopping, Grad-CAM check, ONNX export).
Datasets go in backend/training/data/ (gitignored). Add tests for the split and metric code. Stop and report.
```

**T3: train one tabular model** (repeat per repo)
```
Train <REPO> with backend/training/ following docs/TRAINING_PLAN.md Track 2 and the standard recipe.
Use the dataset recorded in docs/TRAINING_QUEUE.md (tell me if it needs my login). Reproduce the repo's
preprocessing. Report test-set accuracy, sensitivity, specificity and AUC, and compare them with what the repo
claims. If it meets the bar (sensitivity ≥ 0.80, AUC ≥ 0.80), export to ONNX and integrate following
docs/WORKFLOW.md A6 to A10, with trained_on and the credit line. If not, keep it backlog and explain why.
Stop and report.
```

**T4: prepare one image model** (repeat per repo)
```
Prepare <REPO> for training on Kaggle/Colab following docs/TRAINING_PLAN.md Track 3.
Copy the image template to training/notebooks/<id>.ipynb and fill it in: dataset (Kaggle dataset
slug or download cell), classes, the repo's preprocessing, image size, and the expected outputs
(model.onnx, preprocess.json, metrics.json, gradcam_examples.png). Give me exact step-by-step
instructions to run it on Kaggle with a GPU. Stop and wait. When I give you the downloaded outputs,
check the metrics against the bar and the Grad-CAM examples, then integrate following WORKFLOW.md A6 to A10.
```
