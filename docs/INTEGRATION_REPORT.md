# Integration report: the 126 model repos

Phase 3 of `docs/MASTER_PROMPT.md` (`docs/CLONE_ALL_PROMPT.md`, `docs/WORKFLOW.md` A1–A2). Written 2026-11-04.

**How this was produced.** Every repo was shallow-cloned (`git clone --depth 1`, Git LFS skipped except weight files, Git hooks disabled) into `backend/vendor/<model_id>/` by `backend/tools/clone_repos.py`. The repos are untrusted: nothing from them was run, imported or installed, and no model file was loaded. File facts (commit, size, weight files with SHA-256, licence file, frameworks, external weight links) were read by the script and copied into each registry entry's `audit` field by `backend/tools/update_registry_audit.py`. The 23 wave-1/2 repos were then inspected by reading their code, notebooks (as text) and READMEs. Everything that needs the model to be loaded (exact class order stored inside a pickle, input names of an ONNX graph, feature lists inside a LightGBM file) is marked "confirm in sandbox" and is checked in the integration step (WORKFLOW A4), inside Docker with `--network none`.

**Numbers in READMEs are the authors' claims.** None of them is used as a metric. Metrics are measured by us on held-out data during integration (WORKFLOW A8).

**Licences.** The project owner has written permission from every repo author, so a missing licence does not block a repo (`"permission": "granted by author"` is set on every registry entry; store the letters in `docs/permissions/`). GPL/AGPL code (Ultralytics, the YOLOv8 caries repo) stays server-side.

## Verdicts for the 23 wave-1/2 repos

| # | model_id | Worker | Input | Verdict | One-line reason |
|---|---|---|---|---|---|
| 1 | mlmed_torchxrayvision | torch | Chest X-ray | **READY** | pip library; weights from the GitHub release (pin SHA-256); clear preprocessing; 18 findings |
| 2 | moizah1_diabetes_prediction | tabular | Form (8 Pima fields) | **READY** | sklearn pipeline with imputer, reproducible test split (rs 42); recall may miss the 0.80 bar |
| 3 | joystonmenezes_xray_vision | torch (ONNX) | Upper-limb X-ray | **READY** | already ONNX, preprocessing documented; eval needs the MURA set (Stanford registration) |
| 4 | omerkurtulus_skin_disease_classification | tf | Skin photo | **READY** | 5 classes; Lambda layers need unsafe Keras load (sandbox only) |
| 5 | andreygermanov_yolov8_caries_detector | torch | Teeth photo | **READY** | YOLOv8m detector; GPL, server-side only |
| 6 | yahyazuher_ai_liver_disease_diagnosis_system | tabular | Liver function test | **READY (gate model only)** | gate XGBoost on LFT values; fatty-liver = circular label (SKIP), cancer = synthetic data (SKIP), hepatitis C = hospital data |
| 7 | sneha798_heart_disease_prediction_project | tabular | Form (13 UCI heart fields) | **NEEDS WORK** | which pickle/scale is right is unclear; inputs need stress test + angiography |
| 8 | singh2010nidhi_heart_failure_prediction_using_msazure | tabular (ONNX) | Form (12 fields) | **NEEDS WORK + decision** | predicts DEATH of heart-failure patients; uses follow-up time (leakage) |
| 9 | khaled_ghanem_ocular_disease_classification | tf | Fundus photo | **READY (cataract, glaucoma)**; AMD/DR NEEDS WORK | 4 binary models; fundus camera needed (should be tier B) |
| 10 | ghifar_khder_knee_osteoarthritis_classification | tf | Knee X-ray | **READY** | KL grade 0–4; repo ships a held-out test set (441 images) |
| 11 | srishti_chordia_gestational_diabetes_prediction_using_dl | tf | Form (18 pregnancy labs) | **READY** | Keras + saved scaler; reproducible split; FPG partial leakage to check |
| 12 | sina_imaginative_dental_caries_detection | torch | Dental X-ray (bitewing) | **READY** | YOLOv8n-seg; eval set on Roboflow (account needed) |
| 13 | ceren_kaya_boneage_estimation | tf | Child hand X-ray | **READY + decision** | regression in months: the sensitivity/AUC bar doesn't apply |
| 14 | niaid_tb_lesion_cxr_segmentation | torch | Chest X-ray (TB) | **NEEDS WORK** | lung-segmentation submodule not fetched; segmentation → image probability |
| 15 | diabetic_research_group_diabetes_prediction | tabular | NHANES variables | **NEEDS WORK** | candidate columns include the diagnosis itself; confirm feature list |
| 16 | cilab_ufersa_thyroid_disease_ai | tabular | Thyroid labs | **NEEDS WORK** | class-mean imputation + SMOTE before split; scaler not saved |
| 17 | sanyab1801_thyroid_disease_prediction | tabular | Thyroid labs (1983) | **NEEDS WORK** | scaler not saved; needs a TRH stimulation test |
| 18 | sathwikaega_pcos_detection | tabular | 41 fields incl. ultrasound | **NEEDS WORK** | no random seed → no honest held-out set; ultrasound inputs |
| 19 | rash200319_two_stage_preeclampsia_ai_model | tabular | Vitals + history | **NEEDS WORK + decision** | label is "high maternal risk", not preeclampsia; simulated features |
| 20 | seunghahh_predictinghypertension | tabular | Korean NHIS fields | **NEEDS WORK** | no training code / target; Korea-specific codes; BP is an input |
| 21 | ishad05_anemia_ml_project | tabular | CBC | **SKIP** | labels made by a rule on the inputs (circular); app label bug |
| 22 | yudaleng_copd_early_prediction | torch | Spirometer curve | **SKIP for app** | needs a raw spirogram; UK Biobank data → clinic portal (Route E) |
| 23 | madhukalita_health_information_and_diagnostic_system | tabular | Form (stroke) | **SKIP** | the stroke model file is missing from the repo → retrain (Track 2) |

**Totals:** 11 READY (two of them only partly: liver gate model, ocular cataract + glaucoma), 9 NEEDS WORK, 3 SKIP.

**Recommended integration order for Phase 5** (after torchxrayvision in Phase 4): moizah1 diabetes → yahyazuher liver gate → joystonmenezes bone X-ray → omerkurtulus skin → andreygermanov caries → ghifar knee → srishti GDM → khaled ocular (cataract, glaucoma) → sina dental → ceren bone age (after the metric decision), then the NEEDS WORK repos.

## Decisions

On 2026-11-04 the project owner asked for the recommended option to be applied for each open question. These are the decisions; any of them can be changed later.

1. **Heart failure (singh2010nidhi): not shown to app users as-is.** The model predicts whether a heart-failure patient dies during follow-up, using the follow-up length as an input (leakage). In Phase 5 it is retrained without `time` (Track 2 recipe) and, if it passes the bar, offered only in the clinic portal (Route E) as clinician information. Patients never see a "risk of dying" score.
2. **Bone age (ceren_kaya): acceptance bar = mean absolute error ≤ 12 months on held-out RSNA images** (sensitivity/AUC don't apply to a regression model). It is offered as an X-ray reading aid for a parent with the child's X-ray report, worded "estimated bone age", and always "discuss with your paediatrician".
3. **Preeclampsia (rash200319): integrated only under its true name, "maternal health risk (high vs not high)"**, with the simulated environment features removed and the model retrained without them; never called preeclampsia. Model B (phenotypes) stays out of the app.
4. **Rule-based lab flags: not shipped as ML.** The anemia (Hb + Mentzer index) and fatty-liver (triglycerides + ALT/GGT) rules may be shown later as deterministic, clearly-labelled lab-report flags with their source; their pickles are not integrated.
5. **Logins for evaluation data** (free, no money): Kaggle (RSNA bone age and others), Stanford MURA registration (bone X-ray), Roboflow (dental #2). Asked for one at a time when that model is reached.
6. **Registry tier: ocular models move from tier A to tier B** (they need a fundus camera image). Changed at integration time (Phase 5), not in this cloning step.
7. **The two repos over 2 GB are kept as metadata only** (`eranzvuloni_msd_generalization` 2.05 GB, `rajpurkarlab_panendosuite` 2.74 GB; both hospital-only, not needed before Phase 8). Listed in `backend/tools/metadata_only_repos.txt`; delete a line and re-run the clone script to download one fully.

## Wave-1 and wave-2: detailed inspection

### moizah1_diabetes_prediction (wave-1, tabular)
- Weights: models/final_model.joblib (CalibratedClassifierCV isotonic cv=5 around Pipeline[KNNImputer(5) -> StandardScaler -> LogisticRegression(C=0.1, lbfgs, l2)]); models/best_pipeline_uncalibrated.joblib (for SHAP); artifacts/shap_explainer.joblib; artifacts/metadata.json (features, medians, ranges).
- Features in order: Pregnancies, Glucose (mg/dL, 2-h OGTT), BloodPressure (diastolic mmHg), SkinThickness (triceps mm), Insulin (2-h serum µU/mL), BMI, DiabetesPedigreeFunction, Age (years).
- Preprocessing: 0 -> NaN for Glucose, BloodPressure, SkinThickness, Insulin, BMI; KNNImputer inside pipeline => unknown fields can be sent as NaN.
- Output: predict_proba[:,1] = P(Outcome=1, diabetes). Labels 0 lower / 1 higher risk.
- Data: data/diabetes_raw.csv = Pima Indians Diabetes, 768 rows (women >=21, Pima heritage). Split: train_test_split test_size=0.2 stratify random_state=42 => test split reproducible for honest eval.
- README claims (NOT ours): LR test AUC 0.84, recall 0.59, acc 0.79. Recall likely < 0.80 bar at 0.5 threshold.
- Framework: scikit-learn >=1.3 (+xgboost, shap in reqs). Worker: tabular. ONNX: skl2onnx supports KNNImputer/StandardScaler/LogReg; CalibratedClassifierCV supported (isotonic? check).
- Licence: none found (README badge says MIT) -> permission granted by author.
- Risks: women-only, Pima population; glucose is OGTT 2-h not fasting; calibration wrapper conversion; sensitivity at 0.5.
- Verdict: READY.

### sneha798_heart_disease_prediction_project (wave-1, tabular)
- Weights: random_forest_model.pkl (1.2 MB; app.py loads this), HEART_MODEL.pkl (9 KB; notebook pickles `adaboost_clf`).
- Features in order (prediction.py): age, sex(1=male), cp(0-3 chest pain type), trestbps (resting BP mmHg), chol (mg/dL), fbs (fasting sugar >120: 1/0), restecg(0-2), thalach (max heart rate), exang (exercise angina 1/0), oldpeak (ST depression), slope(0-2), ca (0-4 vessels by fluoroscopy), thal (0-3).
- Preprocessing: app passes raw values. Notebook builds MinMaxScaler/StandardScaler versions (norm_df_x) and some models trained on scaled data -> must confirm in sandbox which input scale each pickle expects.
- Provenance unclear: random_forest_model.pkl not created in the notebook; TODO.md mentions "RandomForestRegressor ... random_forest_model.pkl".
- Data: heart.csv, 303 rows (UCI Cleveland / Kaggle heart.csv, target 1=disease). Split in notebook: test_size 0.2 stratify random_state 42.
- Licence: MIT. Worker: tabular.
- Risks: ca, thal, oldpeak, slope, restecg need a hospital ECG/stress test/angiography: a phone user rarely knows them. Tiny dataset.
- Verdict: NEEDS WORK: confirm model class + expected scaling in sandbox; inputs need clinical test results (form must say so).

### yahyazuher_ai_liver_disease_diagnosis_system (wave-1, tabular; 6 models in one repo)
- Weights (models/): gate_model.pkl, fatty_liver_model.pkl, cancer_model.pkl, hepatitisC_stage_model.pkl, hepatitisC_status_model.pkl, hepatitisC_complications.pkl. All XGBClassifier (xgboost) pickled with joblib/pickle.
- Gate model (the phone-usable one: liver function test report): features in order Age, Gender (Male=1, Female=0), Total_Bilirubin (mg/dL), Direct_Bilirubin (mg/dL), ALP (IU/L), ALT (IU/L), AST (IU/L), Total_Protiens (g/dL), Albumin (g/dL), Albumin_and_Globulin_Ratio. No scaling (trees). Target: 0 = liver patient ("Sick"), 1 = healthy (dataset 1->0, 2->1). XGB scale_pos_weight 1.5. Split test_size 0.2 stratify rs 42.
- Gate data: Liver_Patient_Dataset_Cleaned_19k.csv (Kaggle "Liver Disease Patient Dataset" 30k LPD; ILPD-derived, India). Risk: this Kaggle set is an augmented/duplicated ILPD; duplicates across train/test inflate metrics -> evaluate on the original ILPD (UCI, 583 rows) de-duplicated against training rows.
- Fatty liver model: label = rule on its own inputs (TG>150 & (ALT>40 or GGT>40)) or (ALT>40 & GGT>40) from NHANES 2013-14 labs => circular; not a NAFLD model. Do not integrate as ML; at most show the rule as a lab-flag.
- Cancer model: The_Cancer_data_1500 (Kaggle rabieelkharoua "cancer prediction dataset") - synthetic tabular data; not clinically valid.
- Hepatitis C stage/status/complications: Mayo PBC cirrhosis dataset (418 patients) - hospital labs + histology; not phone-usable.
- Licence: MIT. Worker: tabular (xgboost -> onnxmltools).
- Verdict: gate model READY (needs held-out eval on original ILPD with dedup); fatty liver SKIP (circular label); cancer SKIP (synthetic data); hepatitis C models SKIP for app (hospital data).

### singh2010nidhi_heart_failure_prediction_using_msazure (wave-1, tabular)
- Weights: best_model.onnx (497 KB, Azure AutoML export with enable_onnx_compatible_models), automl-model.pkl (2.3 MB, AutoML pipeline needs azureml SDK).
- Features (12, CSV order): age, anaemia (0/1), creatinine_phosphokinase (mcg/L), diabetes (0/1), ejection_fraction (%), high_blood_pressure (0/1), platelets (kiloplatelets/mL), serum_creatinine (mg/dL), serum_sodium (mEq/L), sex (1=male), smoking (0/1), time (follow-up days).
- Target: DEATH_EVENT (death during follow-up). Data: Heart Failure Clinical Records (Chicco & Jurman 2020; 299 heart-failure patients, Faisalabad, Pakistan, 2015).
- Problems: (1) `time` = length of follow-up, unknown at prediction time -> target leakage; (2) the task is MORTALITY of existing heart-failure patients, not heart-failure risk for app users; (3) ejection fraction, CPK need echo/labs. ONNX input names/featurizer must be checked in the sandbox.
- Licence: none found -> permission granted by author. Worker: tabular (ONNX runtime).
- Verdict: NEEDS WORK + product decision: showing a "risk of dying" result to app users; would need retraining without `time`.

### joystonmenezes_xray_vision (wave-1, image upload: bone X-ray)
- Weights: model/densenet121_mura.onnx (DenseNet-121, original 2021 TF weights converted to ONNX; ImageNet BGR mean subtraction is inside the graph).
- Input: one upper-limb radiograph (elbow, finger, forearm, hand, humerus, shoulder, wrist). Preprocessing (app/preprocessing.py): RGB, resize longest side to 256 keeping aspect, pad to 256x256 black, centre-crop 224, /255, NHWC float32 [1,224,224,3].
- Output: 2-class softmax (normal, abnormal); app averages TTA flips. Saliency: input-gradient map (not Grad-CAM).
- Data: Stanford MURA (40,561 images, 14,863 studies, upper extremity only). Eval needs the MURA validation set: requires a free Stanford registration (user's login) -> ask user.
- Licence: MIT (code). MURA data is research-use only.
- Worker: torch worker (ONNX runtime is already in worker-torch reqs) - no torch needed for inference.
- Risks: upper extremity only (no leg/spine/chest); a phone photo of a screen degrades accuracy; README accuracy is from 2021, not ours.
- Verdict: READY.

### omerkurtulus_skin_disease_classification (wave-1, phone photo: skin)
- Weights: models/Model_10_SalvationNet_best.keras and _final.keras (9.7 MB each; Keras 3 format). Only SalvationNet (from-scratch net) is shipped.
- Input: RGB photo, tf.image.resize to IMG_SIZE_SALVATION (README: 224x224x3); normalisation happens INSIDE the model (Lambda z/255). Model uses several Python Lambda layers (attention/pooling) -> loading needs safe_mode=False (deserialises Python bytecode: arbitrary code) -> sandbox only, then tf2onnx.
- Classes (5, output order from notebook CLASS_NAMES): acne, hyperpigmentation, Nail_psoriasis, SJS-TEN, Vitiligo.
- Data: Khushbu 2024 "Skin Disease Classification Dataset", Mendeley Data DOI 10.17632/3hckgznc67.1, CC BY 4.0; 3,411 photos after removing augmented duplicates; hospitals + online sources; skin-tone mix not documented.
- Licence: MIT. Worker: tf.
- Risks: Lambda layers + Python-version-bound bytecode may fail to load (45-min rule); only 5 classes -> any other rash is forced into one of them (needs an "out of scope" warning); SJS-TEN is a medical emergency -> a positive must escalate.
- Verdict: READY (eval possible on the Mendeley set, CC BY 4.0, but note the model saw part of it in training -> use the repo's documented test split if reproducible).

### andreygermanov_yolov8_caries_detector (wave-1, phone photo: teeth)
- Weights: best.pt (52 MB, YOLOv8m fine-tuned 30 epochs; Ultralytics checkpoint = pickle -> sandbox, then `model.export(format="onnx")`).
- Input: RGB photo of teeth (intra-oral). Ultralytics default letterbox to 640, /255. Detector, not classifier: returns boxes with class + confidence.
- Classes: from the DentalAI dataset (Supervisely): caries, cavity, crack, tooth (exact names/order are stored in best.pt; confirm in sandbox).
- Mapping for the app: label "Possible caries/cavity/crack spotted" if any non-tooth box >= threshold, probabilities from max box confidence per class; return boxes as the explanation overlay.
- Data: DentalAI (datasetninja.com/dentalai): intra-oral photos; licence of dataset to confirm (listed as CC BY-NC-SA on datasetninja?) -> check before evaluation.
- Licence: GPL-3.0 (ultralytics is AGPL) -> server-side only, never shipped in the app.
- Worker: torch.
- Risks: detection metrics (mAP) need box-level eval; for the 0.80 bar we compute image-level sensitivity/AUC (any caries box vs none). Phone photos differ from the clinical camera images in training.
- Verdict: READY.

### mlmed_torchxrayvision (wave-1, image upload: chest X-ray) — FIRST MODEL (Phase 4)
- Weights: not in the repo (only a 3 MB scripts/*.pkl helper). Library downloads `densenet121-res224-all` from the GitHub release on first use (model_urls in torchxrayvision/models.py) -> fetch once, pin SHA-256, bake into the worker image / volume so it runs offline.
- Input: single frontal chest X-ray. Preprocessing (README): grayscale; xrv.datasets.normalize(img, 255) -> [-1024, 1024]; XRayCenterCrop; XRayResizer(224); tensor [1,1,224,224].
- Output: 18 sigmoid scores (model.pathologies), e.g. Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum. The `all` weights ship op_threshs (per-pathology operating points); the library rescales outputs so 0.5 = operating point. Also ppv80_thres.
- Data: NIH ChestX-ray14, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, Kaggle RSNA (the "all" model).
- Licence: Apache-2.0 for the library (LICENSE notes baseline models vary).
- Worker: torch (pip torchxrayvision; Grad-CAM via pytorch-grad-cam on features).
- Eval candidates (held-out, not in "all" training): e.g. a public pneumonia set not in the list (Kermany pediatric CXR is different population) -> decide in Phase 4.
- Risks: phone photos of film/monitors; lateral views; children. Non-chest images must be rejected.
- Verdict: READY.

### ishad05_anemia_ml_project (wave-2, tabular: CBC report)
- Weights: anemia_model.pkl and model/anemia_model.pkl (joblib, sklearn; notebook uses StandardScaler, LabelEncoder, SMOTE).
- Features in order (anemia_app.py): RBC (million/µL), PCV (%), MCV (fL), MCH (pg), MCHC (g/dL), RDW (%), TLC (cells/mm³), "PLT /mm3", HGB (g/dL), mentzer_index (= MCV/RBC).
- Labels: created by a RULE on the inputs: HGB < 12 (F) / 13 (M) -> anemic; then Mentzer index > 13 -> "Iron Deficiency Anemia", else "Thalassemia Trait"; else "Non Anemic". => CIRCULAR: the model re-learns its own rule; no lab-confirmed diagnosis (no ferritin / HbA2 electrophoresis).
- Bug: LabelEncoder sorts classes alphabetically (0 Iron Deficiency Anemia, 1 Non Anemic, 2 Thalassemia Trait) but anemia_app.py shows {0: Normal, 1: IDA, 2: Thalassemia} -> its displayed labels are wrong.
- Data: data/cbc_anemia.csv, 373 CBC rows (Indian hospital source not documented; README says India context).
- Licence: none found -> permission granted by author. Worker: tabular.
- Verdict: SKIP as an ML model (circular label). The Mentzer-index rule itself is a standard screening heuristic and could be offered as a deterministic, clearly-labelled lab flag (product decision). A real model needs a dataset with confirmed diagnoses (Track 2 retrain).

### sanyab1801_thyroid_disease_prediction (wave-2, tabular)
- Weights: thyroid_model.pkl (joblib, sklearn HistGradientBoostingClassifier).
- Data: UCI "new-thyroid" (Thyroid gland data, Coomans 1983): 215 patients; Class 1 normal, 2 hyper, 3 hypo. Features in order: Attribute1 T3-resin uptake (%), Attribute2 total serum thyroxin T4 (Isotopic displacement), Attribute3 total serum triiodothyronine T3, Attribute4 basal TSH, Attribute5 max abs. difference of TSH after 200 µg TRH injection.
- Preprocessing: StandardScaler fit on X_train (test_size 0.2, rs 42) and the model is trained on SCALED inputs, but the scaler is NOT saved. It can be reconstructed exactly from the public UCI file with the same split.
- Licence: none found -> permission granted by author. Worker: tabular.
- Risks: Attribute5 needs a TRH stimulation test (rarely done today); 1983 assay units; 215 rows.
- Verdict: NEEDS WORK: rebuild scaler from UCI data + same split; inputs not obtainable by a typical user (TRH test). Better candidate: cilab_ufersa_thyroid_disease_ai.

### sathwikaega_pcos_detection (wave-2, tabular)
- Weights: model.pkl (pickle, XGBClassifier, no scaler).
- Features (41, order in apps.py): age, weight, height, BMI, blood group (code), pulse rate, respiratory rate, Hb, cycle (R/I code), cycle length, marriage years, pregnancies, abortions, I beta-HCG, II beta-HCG, FSH, LH, FSH/LH, hip, waist, waist/hip, TSH, AMH, PRL, Vit D3, PRG, RBS, weight gain, hair growth, skin darkening, hair loss, pimples, fast food, regular exercise, BP systolic, BP diastolic, follicle no. L, follicle no. R, avg follicle size L, avg follicle size R, endometrium thickness.
- Data: Kaggle "PCOS dataset" (Kottarathil 2020; 541 women, 10 hospitals in Kerala, India) = PCOS_data_without_infertility.xlsx + PCOS_infertility.csv merged.
- Split: train_test_split(test_size=0.3) with NO random_state -> the training rows are unknown, so no honest held-out evaluation is possible on this dataset.
- Licence: none found -> permission granted by author. Worker: tabular.
- Risks: 11 inputs need ultrasound (follicle counts/sizes, endometrium) and 9 need hormone labs; only symptom fields are phone-friendly. Repo app says "You are at risk of PCOS disease" (diagnostic wording).
- Verdict: NEEDS WORK: cannot be evaluated honestly (no seed) and needs ultrasound inputs. Track 2: retrain with a fixed split, and consider a symptom + basic-lab model.

### cilab_ufersa_thyroid_disease_ai (wave-2, tabular)
- Weights: thyroid_disease_AI/models_file/*.sav (DecisionTree, ExtraTrees, GradientBoost, LightGBM, RandomCV, StackingModel, XGBoostClassifier) + sys_model_xai/model.sav; joblib.
- Features in order: TT4, TT4 measured (t/f), T4U measured (t/f), T3 measured (t/f), FTI, T3, TSH, T4U, pregnant (t/f), I131 treatment (t/f). Target binaryClass (cat codes: N=0 negative, P=1 hypothyroid).
- Data: UCI Thyroid "hypothyroid" (Garavan Institute, Sydney, 1980s; ~3,163 records).
- Preprocessing problems: (1) missing TT4/FTI/T3/TSH/T4U imputed with the mean OF THEIR OWN CLASS (uses the label -> leakage); (2) SMOTE applied to the WHOLE dataset before the train/test split -> synthetic points in the test set; (3) StandardScaler fit in utils.slipt_and_standardize_dataset and NOT saved (input_test.csv is already scaled). Reported AUC/accuracy are therefore not trustworthy and there is no clean held-out set.
- Licence: MIT. Worker: tabular.
- Verdict: NEEDS WORK: scaler not saved and evaluation leakage; Track 2 retrain on UCI thyroid with a proper split (keep the same features, which ARE on a normal thyroid lab report: TSH, T3, TT4, T4U, FTI).

### seunghahh_predictinghypertension (wave-2, tabular)
- Weights: model.pkl (pickle; class unknown until sandbox load), hypertension_norm.mat (min/max per feature, MATLAB v5; read with scipy.io.loadmat).
- Features in order (sample CSV): SEX, AGE (5-year band midpoint), REGION (code), INCOME (decile code), BP_HIGH (systolic), BP_LWST (diastolic), BLDS (fasting glucose), TOT_CHOLE, HMG (Hb), SGOT_AST, SGPT_ALT, GAMMA_GTP, BMI, HIS_DIABETES, HIS_STROKE, SMK_STAT (1-3), DRINK_FREQ, EXERCI_FREQ. Column names = Korean NHIS health-screening cohort.
- Preprocessing: min-max scaling with the .mat arrays, reshape (1, 18); output predict_proba[:,1] = hypertension.
- No training code, no description of the target, of the dataset version or of the split. Two inputs are the blood pressure itself.
- Data: Korean NHIS (National Health Insurance Service) screening data - access requires a Korean NHIS application -> no held-out set we can get.
- Licence: none found -> permission granted by author. Worker: tabular.
- Verdict: NEEDS WORK: target undefined (likely "already has hypertension" from BP itself = circular), REGION/INCOME are Korea-specific codes, cannot be evaluated. Consider Track 2 with a public dataset instead.

### diabetic_research_group_diabetes_prediction (wave-2, tabular)
- Weights: models/artifacts/model.pkl (MLflow lightgbm flavour, LGBMClassifier, lgb 4.6.0, Python 3.13.6, 5.5 MB). Published: IEEE "Type II Diabetes Risk Prediction: A Multifactor Approach Using Laboratory and Lifestyle Features".
- Target: label_three_class = Not diabetic / T2D / Other (all other diabetes types aggregated). Train set balanced ~4,320 per class; test parquet keeps the original distribution (data/diabetes_test.parquet = a held-out set we CAN use).
- Columns in the parquet (NHANES codes, ~70): demographics (RIDAGEYR, DMARACER...), body (BMXBMI, BMXWAIST, BMXHT, BMXWT), BP (BPXSAR, BPXDAR, pulse), labs (LBXGH HbA1c, LBXGLU glucose, LBDINSI insulin, lipids, creatinine, urine albumin, CRP, B12...), questionnaires (DIQ010 "doctor told you have diabetes", DIQ050 insulin use, DIQ070 diabetes pills, DID040 age at diagnosis, DIQ220, MCQ...), medications (RXDDRUG) and mortality follow-up (MORTSTAT, DIABETES__mortality, VNDIABETES__mortality, VNUCOD_LEADING).
- Leakage: several columns ARE the diagnosis or its treatment (DIQ010, DIQ050, DIQ070, DID040, RXDDRUG, diabetes mortality flags). Which columns the saved model uses must be read from model.feature_name_ in the sandbox; if any of these are inputs, the model is circular for app users.
- Licence: MIT. Worker: tabular (lightgbm -> onnxmltools).
- Verdict: NEEDS WORK: confirm feature list in sandbox; if diagnosis/treatment columns are used, retrain without them (Track 2). The held-out parquet makes honest evaluation possible.

### srishti_chordia_gestational_diabetes_prediction_using_dl (wave-2, tabular)
- Weights: gdm_model.h5 (Keras Sequential, HDF5 legacy; sigmoid output = P(GDM)) + scaler.joblib (StandardScaler).
- Features in order (18; app.py): AP (adverse pregnancy history 0/1), ICP (intrahepatic cholestasis 0/1), TD (thyroid disease 0/1), Eclampsia (0/1), Age, BMI, ALT, AST, GGT, ALP, TBA (total bile acid), UREA, CREA, UA (uric acid), BMG (beta-2 microglobulin), A1MG (alpha-1 microglobulin), CysC (cystatin C), FPG (fasting plasma glucose). Units: SI (mmol/L, µmol/L) - Chinese hospital lab panel; must confirm against data ranges.
- Data: gestational_diabetes.csv, 5,700 pregnancies (early-pregnancy labs; columns also Twins, Day, TP, ALB, TBIL, DBIL, CO2 that the app drops). Source not named in README.
- Split: test_size 0.2 random_state 42 -> reproducible held-out split.
- Licence: none found -> permission granted by author. Worker: tf (h5 -> tf2onnx).
- Risks: FPG is itself one of the GDM diagnostic criteria (partial leakage, depending on when it was measured: check "Day"); population is Chinese; kidney/liver panel inputs need a full lab report.
- Verdict: READY (with the FPG caveat documented and checked during evaluation).

### rash200319_two_stage_preeclampsia_ai_model (wave-2, tabular)
- Weights: models/biofusion_model_v1.pkl (Model A: sklearn Pipeline StandardScaler -> MLPClassifier, grid-searched for recall), models/feature_columns.pkl (expected column order, app reindexes with fill 0), models/preeclampsia_phenotype_model.pkl + cluster_imputer/cluster_scaler/classifier_imputer (Model B).
- Model A inputs: Age, Systolic BP, Diastolic, BS (blood sugar mmol/L), Body Temp (°F), Heart Rate, BMI, Gestational Diabetes, Preexisting Diabetes, Mental Health, Previous Complications (0/1), MAP (derived), Heat_Exposure, Air_Pollution, Access_To_Care (README: "contextual features used to SIMULATE low-resource settings").
- Model A target: "Risk Level" High = 1 vs Mid/Low = 0 from "Dataset - Updated.csv" (1,205 rows; = the UCI/Kaggle Maternal Health Risk data, Bangladesh, with extra columns). It is NOT a preeclampsia label.
- Model B: ~400 clinical/biomarker rows (PlGF/sFlt, sEng, PP-13...) with NO preeclampsia label; target found by clustering -> phenotype, not a diagnosis.
- Licence: MIT. Worker: tabular.
- Verdict: NEEDS WORK: Model A must be presented as "maternal health risk (high vs not)", not preeclampsia, and the simulated environment features must be dropped or fixed; Model B is research/clinician-only. Product decision on naming.

### yudaleng_copd_early_prediction (wave-2, signal: spirogram)
- Weights: weights/SpiroEncoder.pth (PyTorch Net1D/DeepSpiro), weights/SpiroExplainer.cbm and SpiroPredictor.cbm (CatBoost).
- Input: a raw spirometry Volume-Flow time series (xlsx; ./data/sample.xlsx) + age, sex, smoking. Output: COPD detection probability + 1-, 2-, 3-, 4-, 5+-year COPD risk (SpiroPredictor).
- Data: UK Biobank spirometry (access requires an approved UK Biobank application -> no held-out set we can get). Paper: Nature Communications / npj (DeepSpiro).
- Licence: CC (Creative Commons; check variant). Worker: torch (+ catboost; .cbm loads with catboost, .pth is a pickle -> sandbox).
- Risks: phone users don't have a raw spirogram (only hospital/clinic spirometers export curves); no public eval data.
- Verdict: SKIP for the phone app (needs a spirometer curve) -> belongs with the clinic portal (Route E); keep it NEEDS WORK if a spirometry-export upload is added later.

### ghifar_khder_knee_osteoarthritis_classification (wave-2, image upload: knee X-ray)
- Weights: final-model.keras (131.6 MB, Keras 3; stored via Git LFS, pulled).
- Input: knee X-ray -> RGB, resize 224x224 (no aspect keep), /255, [1,224,224,3]. Grad-CAM already implemented in app.py.
- Output: 5-class softmax, Kellgren-Lawrence grade 0-4 (0 healthy, 1 doubtful, 2 minimal, 3 moderate, 4 severe).
- Data: Kaggle "Digital Knee X-ray" (1,650 images, two expert label sets) + OAI-derived Kaggle set (pre-split 70/10/20); class balancing with a GAN (synthetic training images).
- Test-data/ in repo = their held-out test images per KL grade -> usable for our evaluation (confirm no overlap with GAN-generated train images).
- Licence: none found -> permission granted by author. Worker: tf.
- Risks: GAN-synthesised training images; X-ray must be a single knee AP view (the repo splits bilateral images); 224 squash distorts aspect.
- Verdict: READY.

### khaled_ghanem_ocular_disease_classification (wave-2, image upload: retinal fundus)
- Weights (4 separate Keras HDF5 models; file names carry the authors' accuracy, not ours): AMD/saved_model/VGG19-AMD_classification-84.21.h5, cataract/saved_model/ResNet50-Cataract_classification-98.98.h5 (+ a TF SavedModel copy), Diabetic_Retinopathy/saved_model/inception-DR_classification-95.00.h5, glaucoma/saved_model/VGG19-Glaucoma_classification-98.91.h5.
- Input: fundus photograph, cv2 (BGR) resize 224x224; per-notebook extra preprocessing (AMD: unsharp mask; DR: BGR + blur/weighted Ben-Graham-style) that must be replicated exactly per model.
- Outputs: AMD, cataract, glaucoma = binary sigmoid (normal vs disease). DR: InceptionV3 head (Dense 256 ...) - number of output classes to confirm in sandbox (APTOS grades 0-4 or binary).
- Data: ODIR-5K (AMD, cataract; Shanggong Medical, China), APTOS 2019 (DR; Aravind Eye Hospital, India), Kaggle "Glaucoma Detection" (sshikamaru). AMD/cataract subsets of 250 per class (sample random_state 42).
- Licence: none found -> permission granted by author. Worker: tf (one adapter per disease = 4 adapters).
- Risks: needs a FUNDUS camera image (registry says tier A, should be B: a phone selfie of the eye won't work); tiny training subsets (250/class); glaucoma Kaggle set has known duplicates; filenames' accuracies are not measured by us.
- Verdict: READY for cataract and glaucoma first (binary, simple preprocessing); AMD and DR NEEDS WORK (preprocessing detail + DR output shape). Registry tier should be B.

### ceren_kaya_boneage_estimation (wave-2, image upload: child hand X-ray)
- Weights: weights/bone_age_custom_cnn.keras (multimodal: image + sex), weights/bone_age_image_only.keras.
- Input: left-hand radiograph, cv2.IMREAD_GRAYSCALE, resize 128x128, /255, [1,128,128,1]; split make_splits val 0.15 seed 42 (stratified by age bins; the RSNA validation set is separate); plus male = 1.0/0.0 for the multimodal model (inputs {"image", "male"}).
- Output: REGRESSION, bone age = output * 240 months (0-20 years). Not a classifier.
- Data: RSNA Pediatric Bone Age Challenge (Kaggle kmader/rsna-bone-age; 12,611 training images). Download needs a Kaggle account (user's login).
- Licence: none found -> permission granted by author. Worker: tf.
- Risks: the sensitivity/AUC >= 0.80 switch-on bar does not fit a regression model -> needs a different bar (e.g. MAE in months) = product decision for the user; paediatric use (children's X-rays).
- Verdict: READY (technically), but needs the user's decision on the acceptance metric (MAE) and on offering a paediatric tool.

### niaid_tb_lesion_cxr_segmentation (wave-2, image upload: chest X-ray, TB)
- Weights (Git LFS, pulled): segment_tb_cxr/unet_resnet18/weights/customunet.pt (UNet, ResNet18 encoder), segment_tb_cxr/yolov8/weights/yolov8.pt (YOLOv8m-seg), segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth (+ plans.json, dataset.json).
- Task: pixel-level TB lesion segmentation on frontal CXR; image-level "TB vs not TB" probability computed from the pixel probability map (classification_tb_not_tb/compute_probability_of_TB_from_segmentation.py; threshold 0.5; ensemble script for nnUNet + YOLOv8).
- Missing: git submodule classification_tb_not_tb/lung_cxr_segmentation (SSH URL git@github.com:niaid/lung_cxr_segmentation.git) was not fetched by the shallow HTTPS clone -> needed for lung masking in the classification step.
- Data: NIAID TB Portals (annotated subset) - access requires a data-use agreement. Public eval candidates: Shenzhen and Montgomery County TB CXR sets (NLM, public).
- Licence: Apache-2.0. Worker: torch (segmentation_models_pytorch, ultralytics, nnunetv2 -> heavy; start with UNet-ResNet18 or YOLOv8-seg alone).
- Risks: overlaps with torchxrayvision routing (both "chest X-ray"); TB is a notifiable disease -> positive result wording must push testing (sputum/CBNAAT).
- Verdict: NEEDS WORK: fetch the lung-segmentation submodule over HTTPS, then integrate the UNet/YOLO path first.

### sina_imaginative_dental_caries_detection (wave-2, image upload: dental bitewing X-ray)
- Weights: best.pt (YOLOv8n-seg, Ultralytics checkpoint = pickle -> sandbox, then ONNX export).
- Input: bitewing or periapical dental radiograph (not a phone photo of teeth). Output: masks + boxes, classes caries and enamel (app uses conf 0.15). Data: Roboflow Universe sina-us3z2/caries-5bj28-r0zdf, 1,653 images, CC BY 4.0 (Roboflow download needs an account/API key).
- README claims mAP50 0.835 on 1,653 annotated X-rays (theirs, not ours).
- Licence: none found in repo (README badge MIT) -> permission granted by author; ultralytics AGPL -> server-side only.
- Worker: torch.
- Risks: users need their dentist's bitewing X-ray; detection metric -> we evaluate image-level caries vs none.
- Verdict: READY (needs a held-out labelled bitewing set; see dataset section).

### madhukalita_health_information_and_diagnostic_system (wave-2, tabular: stroke risk)
- Repo is a multi-model Streamlit app. Weights present: Heart_attack_model.pkl, Heart_attack_model_KNN.pkl, diabetes_model_RandomForest.pkl, cnn_tuberculosis_model.h5, Symptoms_model.pkl (+ LabelEncoder.pkl, labelEncoder_for_Symptoms.pkl, symptoms_list.pkl).
- The STROKE model the registry entry is about (app.py loads "Stroke_rf_model.pkl") is NOT in the repository -> the integration target has no weights.
- Stroke inputs (app.py): gender, age, hypertension, heart disease, ever married, smoking status, work type, residence type, avg glucose level, BMI = Kaggle "Stroke Prediction Dataset" (fedesoriano, 5,110 rows) layout -> a Track 2 retrain is straightforward and phone-friendly.
- README accuracies (97%/95%/85%/95%, "symptoms 100%") are claims, not measured; "100%" suggests a leaky symptom dataset.
- Licence: none found -> permission granted by author. Worker: tabular (+ tf for the TB CNN).
- Verdict: SKIP as-is (stroke weights missing); move stroke to backlog Track 2 (retrain on the public Kaggle stroke dataset with the same features). The other models in this repo duplicate better wave-1/2 candidates.

## Summary of all 126 repos

| model_id | wave | tier | cloned | size (MB) | weights in repo | licence | frameworks |
|---|---|---|---|---|---|---|---|
| andreygermanov_yolov8_caries_detector | wave-1 | A | yes | 49.8 | yes (1) | GPL-3.0 | ultralytics |
| joystonmenezes_xray_vision | wave-1 | B | yes | 33.7 | yes (1) | MIT | onnx, pytorch, tensorflow |
| mlmed_torchxrayvision | wave-1 | B | yes | 61.1 | yes (1) | Apache-2.0 | pytorch, sklearn, tensorflow |
| moizah1_diabetes_prediction | wave-1 | A | yes | 0.5 | yes (3) | none found | sklearn, xgboost |
| omerkurtulus_skin_disease_classification | wave-1 | A | yes | 23.2 | yes (2) | MIT | sklearn, tensorflow |
| singh2010nidhi_heart_failure_prediction_using_msazure | wave-1 | A | yes | 11.1 | yes (2) | none found | onnx, sklearn, tensorflow, xgboost |
| sneha798_heart_disease_prediction_project | wave-1 | A | yes | 2.2 | yes (2) | MIT | sklearn |
| yahyazuher_ai_liver_disease_diagnosis_system | wave-1 | A | yes | 9.4 | yes (6) | MIT | sklearn, xgboost |
| ceren_kaya_boneage_estimation | wave-2 | B | yes | 4.9 | yes (2) | none found | sklearn, tensorflow |
| cilab_ufersa_thyroid_disease_ai | wave-2 | A | yes | 30.8 | yes (8) | MIT | lightgbm, sklearn, xgboost |
| diabetic_research_group_diabetes_prediction | wave-2 | A | yes | 8.5 | yes (1) | Apache-2.0 | lightgbm, sklearn |
| ghifar_khder_knee_osteoarthritis_classification | wave-2 | B | yes | 136.2 | yes (1) | none found | pytorch, sklearn, tensorflow |
| ishad05_anemia_ml_project | wave-2 | A | yes | 1.0 | yes (2) | none found | sklearn |
| khaled_ghanem_ocular_disease_classification | wave-2 | A | yes | 473.8 | yes (6) | none found | sklearn, tensorflow |
| madhukalita_health_information_and_diagnostic_system | wave-2 | A | yes | 19.2 | yes (8) | none found | lightgbm, sklearn, tensorflow, xgboost |
| niaid_tb_lesion_cxr_segmentation | wave-2 | B | yes | 1452.0 | yes (3) | Apache-2.0 | monai, pytorch, sklearn, ultralytics |
| rash200319_two_stage_preeclampsia_ai_model | wave-2 | A | yes | 3.2 | yes (7) | MIT | sklearn |
| sanyab1801_thyroid_disease_prediction | wave-2 | A | yes | 0.3 | yes (1) | none found | sklearn |
| sathwikaega_pcos_detection | wave-2 | A | yes | 3.9 | yes (1) | none found | sklearn, xgboost |
| seunghahh_predictinghypertension | wave-2 | A | yes | 0.3 | yes (1) | none found | ? |
| sina_imaginative_dental_caries_detection | wave-2 | B | yes | 11.0 | yes (1) | none found | ultralytics |
| srishti_chordia_gestational_diabetes_prediction_using_dl | wave-2 | A | yes | 4.5 | yes (2) | none found | sklearn, tensorflow |
| yudaleng_copd_early_prediction | wave-2 | A | yes | 5.7 | yes (3) | CC | catboost, pytorch |
| anchor_neuro_retguard | backlog | B | yes | 6.6 | no | CC | onnx, timm |
| bhuvanaa28_sentiment_analysis_for_depression_detection | backlog | A | yes | 52.8 | no | none found | sklearn |
| casalab_unisa_dino2_dr | backlog | B | yes | 43.7 | no | Apache-2.0 | pytorch, sklearn |
| darwinai_tuberculosisnet | backlog | B | yes | 2.3 | no | GPL-3.0 | sklearn, tensorflow |
| deettoh_skin_lesion_classification | backlog | B | yes | 16.5 | no | MIT | pytorch, sklearn, timm |
| djparente_uti_ml | backlog | A | yes | 0.1 | no | BSD | r-lang |
| duggineniakhil_oral_cancer_detection | backlog | A | yes | 4.7 | no | none found | pytorch, sklearn |
| eeman1113_dental_study | backlog | A | yes | 1734.6 | yes (13) | none found | pytorch, sklearn, ultralytics |
| holy_mahdi_ibids_xray_fracture_detection | backlog | B | yes | 11.8 | no | none found | pytorch, ultralytics |
| ifyoungnet_medgdm | backlog | A | yes | 0.5 | yes (1) | GPL-3.0 | ? |
| lanagarmire_pe_delivery | backlog | A | yes | 1.4 | no | none found | r-lang, sklearn |
| li_xirong_mmc_amd | backlog | B | yes | 12.0 | no | none found | pytorch, sklearn, tensorflow |
| m_aljasem_pediatric_radiology_ai | backlog | B | yes | 0.1 | no | MIT | sklearn, tensorflow |
| marcoparola_detection_framework | backlog | A | yes | 7.8 | yes (1) | MIT | pytorch, sklearn, tensorflow, timm, ultralytics |
| marcoparola_oral_segmentation | backlog | A | yes | 0.2 | no | MIT | pytorch, sklearn |
| nawaz127_a_multilayer_classification_framework_for_skin_disease | backlog | A | yes | 12.8 | no | none found | pytorch, sklearn, timm |
| nyukat_breast_cancer_classifier | backlog | B | yes | 236.9 | yes (1) | AGPL | pytorch, tensorflow |
| nyukat_gmic | backlog | B | yes | 411.9 | yes (1) | AGPL | pytorch, tensorflow |
| pingjunchen_gradingkneeoa | backlog | B | yes | 0.3 | no | MIT | pytorch, sklearn |
| pradamehavannan_oct_based_retinal_disease_classification_and_ai_clinical_decision_support | backlog | B | yes | 1.4 | no | none found | pytorch, sklearn, timm |
| praise_lab_repository_peprml | backlog | A | yes | 12.6 | no | none found | sklearn, tensorflow, xgboost |
| resplab_accept | backlog | A | yes | 8.5 | no | none found | r-lang |
| rick0110_cervical_cancer | backlog | A | yes | 15.5 | no | MIT | pytorch, sklearn, timm |
| sakh1075_multiclass_prediction_of_thalassemia_disorders_using_machine_learning_models_on_cbc_parameters | backlog | A | yes | 30.4 | no | none found | lightgbm, sklearn, xgboost |
| sathvika_09_tb_detection | backlog | B | yes | 0.1 | no | none found | sklearn, tensorflow |
| sfu_mial_skin_iav | backlog | B | yes | 1.0 | no | MIT | monai, pytorch, sklearn |
| shaonint_glaucoma_detection | backlog | B | yes | 7.0 | no | none found | pytorch, sklearn, timm |
| sibisai_skin_scan | backlog | A | yes | 4.6 | no | none found | pytorch, sklearn |
| smon2006_skin_disease_detection | backlog | A | yes | 2.6 | no | none found | pytorch, sklearn |
| uxlabspk_neurovision | backlog | A | yes | 0.0 | yes (1) | MIT | sklearn, tensorflow |
| vankudoth_sunil_kumar_diseaseprediction | backlog | A | yes | 2.1 | yes (9) | none found | sklearn, xgboost |
| vineet416_chronic_kidney_disease_prediction | backlog | A | yes | 1.9 | no | none found | sklearn, xgboost |
| whitedevil71_fundusnext | backlog | B | yes | 0.0 | no | none found | pytorch |
| yehonatanke_thyroid_predict_ml | backlog | A | yes | 8.4 | no | MIT | sklearn, tensorflow |
| aabo5_malaria_detection | hospital-only | C | yes | 57.6 | no | none found | sklearn, tensorflow |
| aarjcode_uti_machinelearning_hai_vs_cai | hospital-only | C | yes | 0.2 | no | none found | sklearn, tensorflow, xgboost |
| amiraynede_lc25000_cancer_classification | hospital-only | C | metadata only | 28.3 | yes (2), 2 LFS pointer | CC | pytorch, sklearn |
| aramis_lab_ad_dl | hospital-only | C | yes | 2.5 | no | MIT | pytorch, sklearn |
| arinova2701_sickleshield | hospital-only | C | yes | 45.3 | yes (1) | none found | pytorch, sklearn |
| bbleier_multimodal_icu_mortality | hospital-only | C | yes | 5.9 | no | MIT | pytorch, sklearn, tensorflow |
| bcv_uniandes_cardium | hospital-only | C | yes | 20.8 | no | Apache-2.0 | pytorch, sklearn, timm |
| bimcv_csusp_bimcv_prostate_classification | hospital-only | C | yes | 277.8 | no | Apache-2.0 | monai, pytorch, sklearn |
| biomedia_mbzuai_fetalclip | hospital-only | C | yes | 5.5 | yes (1) | CC | pytorch, sklearn, timm |
| blackcatdisha_episentinel | hospital-only | C | yes | 95.2 | yes (2) | none found | catboost, lightgbm, sklearn, xgboost |
| boransekeroglu_osteo_models | hospital-only | C | yes | 38.2 | yes (4) | CC | tensorflow |
| conze_slicerpolycystickidneyseg | hospital-only | C | yes | 1.3 | no | Apache-2.0 | pytorch, timm |
| czifan_multimodal_medicine_ai | hospital-only | C | yes | 0.1 | no | none found | ? |
| darshan0109_all_leukemia_detection | hospital-only | C | yes | 3.6 | no | MIT | pytorch, sklearn, timm |
| darshan_cpp_neuroml_ptsd | hospital-only | C | yes | 17.2 | no | MIT | catboost, lightgbm, sklearn, xgboost |
| dayu_lab_db_epifm | hospital-only | C | yes | 0.8 | no | MIT | pytorch, sklearn |
| diagnijmegen_panorama_baseline | hospital-only | C | yes | 1.6 | no | Apache-2.0 | pytorch, sklearn |
| dreamxfar_pos4chd | hospital-only | C | yes | 8.7 | no | none found | pytorch, sklearn, timm, ultralytics |
| eranzvuloni_msd_generalization | hospital-only | C | metadata only | 2054.6 | no | CC | ? |
| esandml_fcbformer | hospital-only | C | yes | 3.1 | no | Apache-2.0 | pytorch, sklearn, timm |
| ezequieldlrosa_deepisles | hospital-only | C | yes | 12.6 | no | Apache-2.0 | monai, pytorch, sklearn |
| fredyandres73_pressure_injury_ml_prediction_code | hospital-only | C | yes | 0.5 | no | none found | ? |
| gatoraim_bat_aki | hospital-only | C | yes | 0.3 | no | none found | pytorch, sklearn, xgboost |
| glydi_i2ireg_clfnet | hospital-only | C | yes | 85.4 | yes (2) | none found | pytorch, sklearn |
| gwadesteve_mttl_research_malaria | hospital-only | C | yes | 7.7 | no | MIT | pytorch, sklearn, ultralytics |
| harish_kush_ocd_classifier | hospital-only | C | yes | 9.1 | yes (4) | none found | sklearn, tensorflow |
| hjiangaz_smart_ccs | hospital-only | C | yes | 25.6 | no | GPL-3.0 | pytorch, sklearn, timm |
| horus_bot_swasthya_predective_system | hospital-only | C | yes | 55.5 | yes (8) | none found | onnx, sklearn |
| hunain6_dengue_risk_prediction_system | hospital-only | C | yes | 0.1 | yes (1) | none found | sklearn |
| ivadomed_seg_sc_ms_lesion_multicontrast | hospital-only | C | yes | 0.5 | no | MIT | monai, pytorch, sklearn |
| jab7_lung_cancer_classification_xai | hospital-only | C | yes | 6.0 | no | none found | sklearn, tensorflow |
| janaristevska_eeg_adhd_detection2 | hospital-only | C | yes | 14.7 | no | Apache-2.0 | pytorch, sklearn, timm |
| kasikrit_ida_thl_classification | hospital-only | C | yes | 0.1 | no | none found | sklearn, tensorflow |
| keerthikkn_hospital_readmission_prediction | hospital-only | C | yes | 13.5 | yes (1) | MIT | sklearn |
| lns98_deep_learning_medical_app | hospital-only | C | yes | 3.5 | no | none found | sklearn, tensorflow |
| lst627_cardiac_fm | hospital-only | C | yes | 2.1 | no | MIT | pytorch, sklearn, tensorflow, xgboost |
| mahmoud_elbattah_glenda_xai | hospital-only | C | yes | 0.0 | no | CC | pytorch, sklearn, timm |
| maynord_7t_ms_lesion_segmentation | hospital-only | C | yes | 5.3 | no | none found | monai, pytorch |
| mazder1_ecg | hospital-only | C | yes | 109.5 | yes (19) | MIT | pytorch, sklearn |
| mesolongo_breast_cancer_classifier_comparison | hospital-only | C | yes | 0.7 | no | none found | sklearn, tensorflow, xgboost |
| mikhailyadgarov_sepsis_prediction | hospital-only | C | yes | 0.5 | yes (1) | MIT | lightgbm, sklearn, xgboost |
| mini_project_sleepap_sleep_apnea_screening_tool | hospital-only | C | yes | 0.1 | no | MIT | pytorch, sklearn |
| mmuratarat_kvasir_v2_vit_classifier | hospital-only | C | yes | 2.3 | no | none found | lightgbm, pytorch, sklearn, tensorflow, xgboost |
| mnaeimaei_parkinson_datscan_classifier | hospital-only | C | yes | 3.0 | no | none found | monai, pytorch, sklearn |
| mohnabilfadl_parkinson_detection_using_voice_analysis | hospital-only | C | yes | 3.5 | yes (1) | none found | sklearn, xgboost |
| mrgiovanni_r_super | hospital-only | C | yes | 29.8 | no | CC | monai, pytorch, sklearn, timm |
| mtichavsky_eeg | hospital-only | C | yes | 13.3 | yes (4) | MIT | pytorch, sklearn |
| muhammad_hassan12_xpathology_colorectal_tissue_specialist | hospital-only | C | yes | 13.0 | no | GPL-3.0 | pytorch, sklearn, timm |
| p_mcc_leukemia_detection | hospital-only | C | yes | 12.2 | no | MIT | sklearn, tensorflow |
| pazjin_fibucl | hospital-only | C | yes | 598.0 | yes (18) | none found | pytorch |
| pgc_ptsd_ewas_ptsd_methylation_risk_score_2024 | hospital-only | C | yes | 21.6 | no | none found | r-lang, sklearn |
| pipi0616_hears_lm | hospital-only | C | yes | 0.2 | no | MIT | lightgbm, pytorch, sklearn |
| project_monai_model_zoo | hospital-only | C | yes | 18.9 | yes (3) | Apache-2.0 | monai, onnx, pytorch, sklearn, tensorflow |
| r1sephoenix_ocd_classification | hospital-only | C | yes | 0.4 | no | none found | lightgbm, pytorch, sklearn |
| rajpurkarlab_panendosuite | hospital-only | C | metadata only | 2736.3 | yes (3) | MIT | ? |
| ronggnahc2026_slor | hospital-only | C | yes | 5.2 | yes (1) | none found | pytorch, sklearn |
| rvandewater_yaib_models | hospital-only | C | yes | 657.1 | yes (93) | MIT | ? |
| sdoerrich97_true_colon | hospital-only | C | yes | 2.5 | no | MIT | pytorch, ultralytics |
| seantangth_cvpr2026_cv4chl | hospital-only | C | yes | 0.6 | no | CC | catboost, lightgbm, pytorch, sklearn, xgboost |
| seherin_zaman_brain_tumor_mri_dataset | hospital-only | C | yes | 23.3 | yes (2) | MIT | sklearn, tensorflow |
| shallencrissle_endoinsights | hospital-only | C | yes | 16.4 | no | none found | pytorch, sklearn, timm |
| socr_pressureinjuryprediction | hospital-only | C | yes | 0.7 | no | LGPL | r-lang |
| subangkar_simsig | hospital-only | C | yes | 188.6 | yes (2) | none found | pytorch, sklearn |
| tabrisrei_isles22_seals | hospital-only | C | yes | 1.4 | no | Apache-2.0 | pytorch, sklearn |
| tadeephuy_cofo | hospital-only | C | yes | 10.3 | no | MIT | pytorch, timm |
| uthmri_deep_learning_ms_segmentation | hospital-only | C | yes | 175.4 | no | GPL-3.0 | tensorflow |
| vaibhav_k_dengue_prediction_ml | hospital-only | C | metadata only | 39.2 | no | none found | sklearn, tensorflow, xgboost |
| vraj_data_scientist_prediction_on_hospital_readmission | hospital-only | C | yes | 5.4 | yes (1) | none found | lightgbm, sklearn, xgboost |
| yangzhou12_openeye_fm | hospital-only | C | yes | 0.1 | no | MIT | pytorch, sklearn, tensorflow, timm |

## Backlog repos (34): why not in the app yet, and what would make them usable

- `anchor_neuro_retguard` (Glaucoma / retinal screening; input: Fundus / OCT): no model weights in the repo; README links weights on github_release. Plan: Track 3: fine-tune image model.
- `bhuvanaa28_sentiment_analysis_for_depression_detection` (Depression text sentiment; input: Text): no model weights in the repo. Plan: Track 4: defer (mental-health text; support routing only).
- `casalab_unisa_dino2_dr` (Diabetic retinopathy; input: Fundus photo): no model weights in the repo. Plan: Track 3: fine-tune image model.
- `darwinai_tuberculosisnet` (Tuberculosis; input: Chest X-ray): no model weights in the repo; README links weights on kaggle. Plan: Track 1: download weights (link in README).
- `deettoh_skin_lesion_classification` (Melanoma / skin lesions; input: Dermoscopy): no model weights in the repo; README links weights on huggingface. Plan: Track 1: download weights (Hugging Face).
- `djparente_uti_ml` (Urinary tract infection; input: Urine / clinical): no model weights in the repo. Plan: Track 2: retrain tabular (urinalysis + symptoms).
- `duggineniakhil_oral_cancer_detection` (Oral cancer; input: Oral photos): no model weights in the repo. Plan: Track 3: fine-tune image model.
- `eeman1113_dental_study` (Dental caries; input: Intraoral photos): weights in repo but not usable as-is; README links weights on huggingface, zenodo. Plan: Track 4: defer (MedGemma LoRA needs a large GPU).
- `holy_mahdi_ibids_xray_fracture_detection` (Bone fractures; input: X-ray): no model weights in the repo. Plan: Track 3: fine-tune image model (YOLO).
- `ifyoungnet_medgdm` (Gestational diabetes; input: Maternal data): weights in repo but not usable as-is. Plan: Track 2: retrain in Python (KNIME model file).
- `lanagarmire_pe_delivery` (Preeclampsia delivery timing; input: Maternal data): no model weights in the repo. Plan: Track 2: retrain tabular (maternal clinical values).
- `li_xirong_mmc_amd` (Age-related macular degeneration; input: OCT / fundus): no model weights in the repo; README links weights on google_drive. Plan: Track 1: download weights (Google Drive).
- `m_aljasem_pediatric_radiology_ai` (Pediatric bone age; input: Hand X-ray): no model weights in the repo. Plan: Track 3: fine-tune image model.
- `marcoparola_detection_framework` (Oral cancer; input: Oral images): weights in repo but not usable as-is. Plan: Track 4: defer (research framework, not one model).
- `marcoparola_oral_segmentation` (Oral lesion segmentation; input: Oral images): no model weights in the repo; README links weights on google_drive, kaggle. Plan: Track 3: fine-tune image model.
- `nawaz127_a_multilayer_classification_framework_for_skin_disease` (Multiclass skin disease; input: Skin photos): no model weights in the repo. Plan: Track 3: fine-tune image model.
- `nyukat_breast_cancer_classifier` (Breast cancer mammography; input: Mammogram): weights in repo but not usable as-is. Plan: Track 1: weights download script in repo.
- `nyukat_gmic` (Breast cancer mammography; input: Mammogram): weights in repo but not usable as-is. Plan: Track 1: weights download script in repo.
- `pingjunchen_gradingkneeoa` (Knee osteoarthritis grading; input: Knee X-ray): no model weights in the repo. Plan: Track 3: fine-tune image model.
- `pradamehavannan_oct_based_retinal_disease_classification_and_ai_clinical_decision_support` (OCT retinal disease; input: OCT): no model weights in the repo; README links weights on github_release, google_drive, kaggle. Plan: Track 3: fine-tune image model.
- `praise_lab_repository_peprml` (Severe preeclampsia; input: Clinical data): no model weights in the repo. Plan: Track 2: retrain tabular (maternal clinical values).
- `resplab_accept` (COPD exacerbation risk; input: Clinical history): no model weights in the repo. Plan: Track 2: port the R model equations, or skip.
- `rick0110_cervical_cancer` (Cervical cancer clinical risk; input: Clinical data): no model weights in the repo. Plan: Track 2: retrain tabular (risk-factor questionnaire).
- `sakh1075_multiclass_prediction_of_thalassemia_disorders_using_machine_learning_models_on_cbc_parameters` (Thalassemia; input: CBC): no model weights in the repo. Plan: Track 2: retrain tabular (CBC).
- `sathvika_09_tb_detection` (Tuberculosis; input: Chest X-ray): no model weights in the repo. Plan: Track 3: fine-tune image model.
- `sfu_mial_skin_iav` (Skin lesion diagnosis; input: Dermoscopy): no model weights in the repo; README links weights on huggingface. Plan: Track 1: download weights (Hugging Face / Zenodo).
- `shaonint_glaucoma_detection` (Glaucoma; input: Fundus): no model weights in the repo. Plan: Track 3: fine-tune image model.
- `sibisai_skin_scan` (Acne / eczema / fungal / psoriasis / scabies; input: Skin photos): no model weights in the repo; README links weights on huggingface, kaggle. Plan: Track 1: download weights (Hugging Face).
- `smon2006_skin_disease_detection` (Acne / eczema / psoriasis / vitiligo / rosacea; input: Skin photos): no model weights in the repo; README links weights on kaggle. Plan: Track 3: fine-tune image model (Kaggle/Colab GPU).
- `uxlabspk_neurovision` (Autism trait screening; input: Questionnaire): weights in repo but not usable as-is. Plan: Track 1 first (.tflite in repo), else Track 2 (questionnaire).
- `vankudoth_sunil_kumar_diseaseprediction` (Liver disease; input: Liver blood tests): weights in repo but not usable as-is; README links weights on kaggle. Plan: Track 2: retrain tabular (liver function tests).
- `vineet416_chronic_kidney_disease_prediction` (Chronic kidney disease; input: Blood / urine): no model weights in the repo. Plan: Track 2: retrain tabular (UCI CKD).
- `whitedevil71_fundusnext` (Glaucoma; input: Fundus): no model weights in the repo; README links weights on huggingface. Plan: Track 1: download weights (Hugging Face).
- `yehonatanke_thyroid_predict_ml` (Hyperthyroidism / hypothyroidism; input: Clinical data): no model weights in the repo. Plan: Track 2: retrain tabular (thyroid labs).

## Hospital-only repos (69): why not in the phone app, and what would make them usable

- `aabo5_malaria_detection` (Malaria): needs Blood smear that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `aarjcode_uti_machinelearning_hai_vs_cai` (Hospital-acquired UTI): needs Clinical records that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `amiraynede_lc25000_cancer_classification` (Lung and colon cancer histology): needs Histopathology that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `aramis_lab_ad_dl` (Alzheimer's disease): needs MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 2D (DICOM/slice).
- `arinova2701_sickleshield` (Sickle cell morphology): needs Blood smear that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `bbleier_multimodal_icu_mortality` (Multimodal ICU mortality): needs ICU multimodal that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `bcv_uniandes_cardium` (Congenital heart defects research): needs Ultrasound that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (specialist imaging).
- `bimcv_csusp_bimcv_prostate_classification` (Prostate cancer classification): needs MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `biomedia_mbzuai_fetalclip` (Fetal ultrasound foundation model): needs Fetal ultrasound that a phone user doesn't have; weights in repo. Plan: Route F: parked foundation model (needs fine-tuning + GPU).
- `blackcatdisha_episentinel` (Dengue outbreak): needs Geographic data that a phone user doesn't have; weights in repo. Plan: Route D: local health alerts (district weather/cases).
- `boransekeroglu_osteo_models` (Osteoporosis): needs MRI / CT that a phone user doesn't have; weights in repo. Plan: Route A: scan upload, 2D (DICOM/slice).
- `conze_slicerpolycystickidneyseg` (Polycystic kidney segmentation): needs CT / MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `czifan_multimodal_medicine_ai` (Ovarian cancer multimodal research, Ovarian cancer research): needs Ultrasound / clinical that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (specialist imaging).
- `darshan0109_all_leukemia_detection` (Leukemia): needs Blood smear that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `darshan_cpp_neuroml_ptsd` (PTSD EEG): needs EEG that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `dayu_lab_db_epifm` (Epilepsy EEG foundation model): needs EEG that a phone user doesn't have; no weights in repo. Plan: Route F: parked foundation model (needs fine-tuning + GPU).
- `diagnijmegen_panorama_baseline` (Pancreatic cancer CT): needs CT that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `dreamxfar_pos4chd` (Congenital heart defects): needs Fetal echo that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (specialist imaging).
- `eranzvuloni_msd_generalization` (Atrial fibrillation / ECG): needs ECG that a phone user doesn't have; no weights in repo. Plan: Route B: wearables / ECG export.
- `esandml_fcbformer` (Colorectal polyp segmentation, Colorectal polyps): needs Colonoscopy that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `ezequieldlrosa_deepisles` (Ischemic stroke lesion segmentation): needs MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `fredyandres73_pressure_injury_ml_prediction_code` (Pressure injury): needs Clinical that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `gatoraim_bat_aki` (Acute kidney injury): needs Clinical time-series that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `glydi_i2ireg_clfnet` (Kidney stone CT research): needs CT that a phone user doesn't have; weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `gwadesteve_mttl_research_malaria` (Malaria): needs Blood smear that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `harish_kush_ocd_classifier` (OCD EEG): needs EEG that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `hjiangaz_smart_ccs` (Cervical cancer cytology, Cervical cancer screening): needs Cytology slide that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `horus_bot_swasthya_predective_system` (Dengue / typhoid / diarrhea outbreak): needs Public health that a phone user doesn't have; weights in repo. Plan: Route D: local health alerts (district weather/cases).
- `hunain6_dengue_risk_prediction_system` (Dengue outbreak): needs Weather / epidemiology that a phone user doesn't have; weights in repo. Plan: Route D: local health alerts (district weather/cases).
- `ivadomed_seg_sc_ms_lesion_multicontrast` (Spinal multiple sclerosis lesion segmentation): needs Spinal MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `jab7_lung_cancer_classification_xai` (Lung cancer pathology): needs Histopathology that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `janaristevska_eeg_adhd_detection2` (ADHD EEG): needs EEG that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `kasikrit_ida_thl_classification` (Iron-deficiency anemia vs thalassemia): needs Blood smear that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `keerthikkn_hospital_readmission_prediction` (Hospital readmission): needs EHR that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `lns98_deep_learning_medical_app` (Crohn's disease imaging): needs MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 2D (DICOM/slice).
- `lst627_cardiac_fm` (Cardiac foundation model): needs ECG / MRI that a phone user doesn't have; no weights in repo. Plan: Route F: parked foundation model (needs fine-tuning + GPU).
- `mahmoud_elbattah_glenda_xai` (Endometriosis lesion recognition): needs Laparoscopy that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `maynord_7t_ms_lesion_segmentation` (Multiple sclerosis 7T lesion segmentation): needs 7T MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `mazder1_ecg` (ECG arrhythmia): needs ECG that a phone user doesn't have; weights in repo. Plan: Route B: wearables / ECG export.
- `mesolongo_breast_cancer_classifier_comparison` (Breast cancer tabular): needs Biopsy features that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `mikhailyadgarov_sepsis_prediction` (Sepsis onset): needs ICU vitals / labs that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `mini_project_sleepap_sleep_apnea_screening_tool` (Sleep apnea): needs Physiological signals that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `mmuratarat_kvasir_v2_vit_classifier` (Esophagitis / GI classification): needs Endoscopy that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `mnaeimaei_parkinson_datscan_classifier` (Parkinsonian disorders DaT SPECT): needs DaT SPECT that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (specialist imaging).
- `mohnabilfadl_parkinson_detection_using_voice_analysis` (Parkinson's voice): needs Voice features that a phone user doesn't have; weights in repo. Plan: Route C: phone voice recording.
- `mrgiovanni_r_super` (Pancreas and kidney tumor segmentation): needs CT that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `mtichavsky_eeg` (Depression / anxiety EEG): needs EEG that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `muhammad_hassan12_xpathology_colorectal_tissue_specialist` (Colorectal tissue classification): needs Histopathology that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `p_mcc_leukemia_detection` (Acute lymphoblastic leukemia): needs Microscopy that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `pazjin_fibucl` (Liver fibrosis): needs Liver MRI that a phone user doesn't have; weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `pgc_ptsd_ewas_ptsd_methylation_risk_score_2024` (PTSD methylation risk): needs DNA methylation that a phone user doesn't have; no weights in repo. Plan: Route G: not feasible (genetic lab data).
- `pipi0616_hears_lm` (Hearing-loss genetic variant interpretation): needs Genetics that a phone user doesn't have; no weights in repo. Plan: Route G: not feasible (genetic lab data).
- `project_monai_model_zoo` (Brain tumor segmentation): needs MRI that a phone user doesn't have; weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `r1sephoenix_ocd_classification` (OCD EEG): needs EEG that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `rajpurkarlab_panendosuite` (GI endoscopic abnormalities): needs Endoscopy that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `ronggnahc2026_slor` (Crohn's treatment response): needs Endoscopy that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `rvandewater_yaib_models` (Acute kidney injury, ICU mortality): needs ICU records that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `sdoerrich97_true_colon` (Colorectal polyp detection): needs Colonoscopy that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `seantangth_cvpr2026_cv4chl` (Cerebral palsy gait): needs Gait pose that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (paediatric gait video).
- `seherin_zaman_brain_tumor_mri_dataset` (Brain tumor classification): needs MRI that a phone user doesn't have; weights in repo. Plan: Route A: scan upload, 2D (DICOM/slice).
- `shallencrissle_endoinsights` (Endometriosis): needs Laparoscopy that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `socr_pressureinjuryprediction` (Pressure injury): needs Clinical that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `subangkar_simsig` (Atrial fibrillation): needs PPG signal that a phone user doesn't have; weights in repo. Plan: Route B: wearables / ECG export.
- `tabrisrei_isles22_seals` (Stroke lesion segmentation): needs MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `tadeephuy_cofo` (Colorectal polyps): needs Colonoscopy that a phone user doesn't have; no weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `uthmri_deep_learning_ms_segmentation` (Multiple sclerosis MRI segmentation): needs MRI that a phone user doesn't have; no weights in repo. Plan: Route A: scan upload, 3D (GPU worker).
- `vaibhav_k_dengue_prediction_ml` (Dengue virus genomic prediction): needs Genomic that a phone user doesn't have; no weights in repo. Plan: Route G: not feasible (virus genomics).
- `vraj_data_scientist_prediction_on_hospital_readmission` (Hospital readmission): needs EHR that a phone user doesn't have; weights in repo. Plan: Route E: clinic portal (doctor uploads the clinical data).
- `yangzhou12_openeye_fm` (Retina foundation model): needs Retina that a phone user doesn't have; no weights in repo. Plan: Route F: parked foundation model (needs fine-tuning + GPU).

## Clone problems

No failures: every repo cloned.
- METADATA ONLY `amiraynede_lc25000_cancer_classification` (28.3 MB in Git): files not checked out: the repo has paths Windows can't create (e.g. ':' in folder names). File list, sizes, licence and weight hashes were read from Git objects.
- METADATA ONLY `eranzvuloni_msd_generalization` (2054.6 MB in Git): kept as metadata only on purpose (listed in metadata_only_repos.txt). File list, sizes, licence and weight hashes were read from Git objects.
- METADATA ONLY `rajpurkarlab_panendosuite` (2736.3 MB in Git): kept as metadata only on purpose (listed in metadata_only_repos.txt). File list, sizes, licence and weight hashes were read from Git objects.
- METADATA ONLY `vaibhav_k_dengue_prediction_ml` (39.2 MB in Git): files not checked out: the repo has paths Windows can't create (e.g. ':' in folder names). File list, sizes, licence and weight hashes were read from Git objects.

## Integration log

- **2026-11-04, `mlmed_torchxrayvision`: integrated, Unverified, switched on by the owner's explicit approval.** ONNX matches the library (max abs diff 1.43e-06). Measured on the Kermany 2018 test split (624 paediatric X-rays, not in training), Pneumonia score at the 0.5 operating point: sensitivity 0.172, specificity 0.991, AUC 0.791, accuracy 0.479, which is below the 0.80 bar. Any lung finding flagged (secondary): sensitivity 0.995, specificity 0.226, AUC 0.913. No adult held-out set without a login was available, so performance on adults (the population it was trained on) is not measured yet.
- **2026-11-04, `mlmed_torchxrayvision` accuracy fix: verified, passes the bar.** Linear layers on the model's features, trained by the Curanex team with validation-only tuning and scored once on held-out data. Adult abnormal X-ray check (VinBigData, n=1035): sensitivity 0.908, specificity 0.904, AUC 0.968. Adult lung opacity check, against normal X-rays and other chest problems: sensitivity 0.925, specificity 0.633, AUC 0.890. Child pneumonia check (Kermany test, n=624): sensitivity 0.944, specificity 0.833, AUC 0.958. A first adult version scored AUC 0.981 against normal X-rays only, but a stress test showed it flagged 78% of other chest problems, so it was replaced. VinBigData is split by image (no patient ids) and its test mix is richer in abnormal X-rays than everyday patients. Results come from Vietnam and China; accuracy in India is not measured.
- **2026-11-04, RSNA adult pneumonia layer: not shipped.** Validation AUC 0.879, but only 0.698 (sensitivity 0.548) on VinBigData, so it did not carry across countries. This is also a warning for the adult lung opacity check, which was trained and tested on one source.
- **2026-11-04, `moizah1_diabetes_prediction`: integrated and verified, passes the bar.** Exported to plain numbers because skl2onnx can't convert the nested KNN imputers (max abs diff vs original 2.2e-16). Held-out split of 154 patients, threshold 0.2007 from training-fold predictions: sensitivity 0.944, specificity 0.56, AUC 0.807, accuracy 0.695. At the default 0.5 threshold: sensitivity 0.519, specificity 0.79.
- **2026-11-04, `yahyazuher_ai_liver_disease_diagnosis_system`: replaced, Unverified.** The repo's gate model failed an external test on UCI HCV data (AUC 0.324). A logistic model trained by the Curanex team on HCV reports scored sensitivity 0.957, specificity 0.963, AUC 0.997 on 183 held-out people, but only sensitivity 0.312, AUC 0.66 on 583 Indian patients (ILPD, GGT missing). Kept switched on as Unverified with that limit stated.
- **2026-11-04, liver update:** retrained on UCI HCV + ILPD together (GGT dropped). Held-out Indian patients (n=171): sensitivity 0.934, specificity 0.224, AUC 0.770. Held-out German patients (n=183): sensitivity 0.652, specificity 0.975, AUC 0.922. Still Unverified.
- **2026-11-04, `omerkurtulus_skin_disease_classification`: backlog.** Moved to backlog on 2026-11-04 (45-minute rule): the .keras file crashes Python (exit 139, segfault in keras load_model), most likely Lambda layer bytecode saved by a different Python version. No test set with a known held-out split either. Candidate replacement: sibisai/skin-scan.
- **2026-11-04, `andreygermanov_yolov8_caries_detector`: integrated and verified.** ONNX export (max abs diff 4.9e-04 on raw output). DentalAI test folder, 250 photos, threshold from the valid folder: sensitivity 0.894, specificity 0.798, AUC 0.929. Not measured on ordinary phone photos.
- **2026-11-04, `ghifar_khder_knee_osteoarthritis_classification`: integrated and verified.** ONNX (max abs diff 1.67e-06). Repo's Test-data, 441 X-rays, grade 2+ vs 0-1 at 0.5: sensitivity 0.809, specificity 0.963, AUC 0.965; exact 5-grade accuracy 0.692, so no exact grade is shown.
- **2026-11-04, `srishti_chordia_gestational_diabetes_prediction_using_dl`: integrated, Unverified.** The .h5 takes 25 inputs while the repo's app sends 18; the scaler was rebuilt from the same split. Held-out real-looking rows: full labs sensitivity 0.878, specificity 0.559, AUC 0.822; only sugar, age and BMI (threshold for sensitivity >= 0.80): sensitivity 0.805, specificity 0.495, AUC 0.752.
- **2026-11-04, `khaled_ghanem_ocular_disease_classification`: backlog (45-minute rule).** Keras 3 could not reload the old models, fixed with the tf-keras legacy loader; the export then timed out at 400 s and was killed for memory (exit 137) under the 3 GB sandbox limit. No ONNX, no metrics, not switched on. Test data, evaluation tool and draft adapter are ready.
- **2026-11-04, `sneha798_heart_disease_prediction_project`: integrated, verified (retrained).** The repo's pickle needs angiography and stress-test inputs and has unclear origin, so a logistic model was trained on patient-answerable questions from four UCI hospitals. Test split: sensitivity 0.855, specificity 0.661, AUC 0.835. Each hospital held out: Cleveland (USA) 0.839, Budapest (Hungary) 0.883, Switzerland 0.784, Long Beach VA (USA) 0.75; sensitivity stayed 0.83 or higher, and two hospitals fell under AUC 0.80.
- **2026-11-04, `cilab_ufersa_thyroid_disease_ai`: integrated, verified (retrained; underactive only).** Official UCI test file, only TSH known: sensitivity 0.873, specificity 0.989, AUC 0.995. With all labs: 0.944 / 0.99 / 0.996. Overactive model failed the bar (sensitivity 0.6, AUC 0.982) and is not used; it was not retuned because the test file was already scored. T3 and T4 are left out of the form because Indian reports use other units.
- **2026-11-04, `rash200319_two_stage_preeclampsia_ai_model`: not switched on.** Retrained on the real columns with duplicates removed: test AUC 0.999, which failed the sense check. Every gestational-diabetes row is labelled High and a depth-3 tree nearly reproduces the label, so the label is close to a rule on the inputs. With only age and blood pressure, sensitivity was 0.0. training/train_maternal.py is kept as the record.
- **2026-11-04, `vineet416_chronic_kidney_disease_prediction`: not switched on.** Sense check before training: every not-CKD row in the UCI data has hemoglobin 13 or above and creatinine 1.2 or below, so a depth-2 tree reaches 0.95 on two values. That is selection in the data, not accuracy, and a model would flag most anemic people. No model trained.
- **2026-11-06, `khaled_ghanem_ocular_disease_classification`: cataract integrated, verified.** Skipping tf2onnx's graph optimizer fixed the memory failure; ONNX matches Keras within 2.4e-07. Independent retina set, half kept for testing: against normal eyes 0.893 / 0.866 / 0.948; against normal plus other eye diseases 0.893 / 0.829 / 0.939. Guard: 80 of 80 retina photos accepted, 80 of 80 X-rays refused, 3 of 40 teeth photos got through. Glaucoma pending.
- **2026-11-06, `khaled_ghanem_ocular_disease_classification`: glaucoma not switched on.** ONNX matches Keras within 5.2e-07, but on the independent retina set it scored AUC 0.566 against normal eyes and 0.474 against everything else, against the repo's 98.91% claim. Half-1 polarity search found no better setting. Possible causes are a preprocessing difference or a model tied to its training source; not retuned because the test half was scored.
- **2026-11-06, `joystonmenezes_xray_vision`: integrated, verified.** MURA validation set from a Kaggle mirror, half of patients held out, per study: sensitivity 0.836, specificity 0.861, AUC 0.903. Weaker parts: elbow 0.758, finger 0.744. MURA's validation set may have guided the original training.
- **2026-11-06, TB chest X-ray layers: not switched on.** Trained on NLM: AUC 0.98 on its own test split, then 0.54 and sensitivity 0.04 on TBX11K. Trained on TBX11K: 5-fold AUC 0.98, then 0.51 on NLM and 0.63 on Belarus plus RSNA. Both learned their source, not TB. The pass rule was fixed in advance and the layer file was removed.
- **2026-11-06, `pamixsun_glaucoma`: integrated, verified (replacement).** Found by web search. Safetensors exported to ONNX in the no-network sandbox, matching within 6e-08. Same independent retina set where the repo model scored AUC 0.47 to 0.57: against normal eyes 0.907 / 0.792 / 0.886; against normal eyes plus other eye diseases 0.907 / 0.752 / 0.869.
- **2026-11-06, `skin_lesion_check`: integrated, Unverified (trained).** HAM10000, split by lesion: sensitivity 0.855, specificity 0.741, AUC 0.882, melanoma found 0.829. Unverified because phone photos were not tested.
- **2026-11-07, `brain_mri_check` and `malaria_cell_check`: integrated, Unverified (trained).** Brain: accuracy 0.929; passed ['meningioma', 'notumor', 'pituitary']; glioma failed on sensitivity. Malaria: accuracy 0.944, sensitivity 0.858, AUC 0.982. LC25000 left out because its images are augmented copies.
- **2026-11-07, `brain_mri_check`: verified.** Outside MRI collection, overlap removed, thresholds unchanged: sensitivity 0.871, specificity 0.812, AUC 0.935 (n=187).
- **2026-11-07, `skin_lesion_check`: failed the smartphone test.** PAD-UFES-20, n=2298: sensitivity 0.548, specificity 0.827, AUC 0.761; melanoma found 0.442. Stays Unverified. Retraining with phone photos.
- **2026-11-07, `skin_condition_check`: integrated, Unverified (trained).** DermNet, 18302 images: 19 of 23 groups passed per class, top-choice accuracy 0.401. Shows three possibilities, and failed groups are hidden.
- **2026-11-07, `malaria_cell_check`: failed the outside test.** BBBC041: sensitivity 0.954, specificity 0.169, AUC 0.661. Stays Unverified.
- **2026-11-07, `skin_lesion_check`: verified after retraining.** Phone photos added to training. Held-out phone patients: sensitivity 0.88, specificity 0.86, AUC 0.943 (n=468); melanoma found 0.625. **`endoscopy_check`: integrated, Unverified.** Polyps and ulcerative colitis passed; esophagitis missed on sensitivity (0.747).
- **2026-11-07, chest X-ray findings added.** VinBigData (Vietnam), threshold on one half, scored once on the other: Cardiomegaly 0.86 / 0.764 / 0.886; Consolidation 0.801 / 0.766 / 0.867; Effusion 0.89 / 0.527 / 0.845.
- **2026-11-07, `oral_condition_check` and `diabetic_retinopathy_check`: integrated, Unverified (trained).** Oral: accuracy 0.933, 6 of 6 classes passed after near-copy removal, image sizes differ by class so source bias is possible; tooth decay left out of the count because the teeth check already covers it. Retinopathy: accuracy 0.79, 5 of 5 grades passed on APTOS.
- **2026-11-07, cancer sets.** `lung_ct_check` and `kidney_ct_check` integrated, Unverified: per-class passes, but slices may share patients across the split. Breast ultrasound failed (malignant sensitivity 0.645) and oral cancer failed (106 images, sensitivity 0.5); both stay off.
- **2026-11-07, outside tests.** `diabetic_retinopathy_check` verified on IDRiD (India): 0.966 / 0.473 / 0.943. `lung_ct_check` failed: specificity 0.0, AUC 0.796. `kidney_ct_check`: locked threshold sensitivity 0.563; threshold from half the outside set, other half 0.891 / 0.946 / 0.976. Breast on BUS-BRA, tested on BUSI: 0.852 / 0.487 / 0.788, not exported.
- **2026-11-07, `kidney_ct_check` correction.** The kidney outside set is largely the training scans resized: a visual-hash check found 82 of 120 normal scans within 3 bits of just 700 sampled training images. Its result is discarded, the threshold is restored, and the model stays Unverified.

## Route C: voice health check (2026-11-10)

`mohnabilfadl_parkinson_detection_using_voice_analysis`: the repo's `model.pkl` was not loaded. It uses RPDE, D2 and PPE, which need research code that a phone app can't reproduce exactly. The Curanex team retrained on the UCI Parkinson's Disease Classification set (Sakar et al. 2018, 252 people, 756 recordings) using only measures that Praat computes from a sustained "aaah" (period, jitter, shimmer, harmonicity, formants, sex). Loudness was left out because it depends on the phone.

- Split by person 60/20/20. Training-fold AUC: logistic 0.736, boosting 0.774, so boosting was used (rule fixed in advance), exported to ONNX (max difference 8e-8).
- Test, scored once: per person (51 people, 35 with Parkinson's) sensitivity 0.886, specificity 0.562, AUC 0.764. Per recording AUC 0.727.
- **Fails the switch-on bar (AUC 0.80).** It was also never tested on phone microphones. It is therefore **Experimental only**: its own screen (`/voice-check`), reached from the Experimental section of the Model Library, never offered to model search or patients' chat.
- Pipeline check: synthetic sustained vowels (webm/opus) go through ffmpeg → Praat → model; junk audio, silence and missing sex are refused with a message. The recording is deleted after measuring.

## Route D: health alert in your area (2026-11-10)

- `hunain6_dengue_risk_prediction_system`: K-means clusters on Pakistan provinces, no outcome to test against. Not used.
- `horus_bot_swasthya_predective_system`: its Chennai data is generated with random numbers (`dataDengue.py`). Not used.
- `blackcatdisha_episentinel`: real weekly dengue counts for 27 Karnataka districts, 2017-2023. Its models need last week's case counts, which no free live source gives, so the repo model was not used. The Curanex team trained a **weather-only** model on its counts (`training/train_dengue_weather.py`), with weather from the Open-Meteo archive so the live inputs (Open-Meteo forecast API, free, no key) are computed the same way.
  - Target (the repo's): next week's cases above the district's 75th percentile. Train 2017-2021, test 2022-2023 once.
  - Year folds: logistic 0.557, boosting 0.596, so boosting (ONNX, SHA-256 pinned). Test: sensitivity 0.713, specificity 0.527, **AUC 0.667**. Weather alone is a weak predictor.
- **What the app shows:** a "Health alert in your area" card on Home for the district in the profile. The headline is measured weather (4-week temperature, humidity, rain) checked against common Aedes ranges, with prevention advice. The model's estimate is only shown under "What to do", marked experimental with its test score, and left out when the weather is outside the Karnataka training range (for example Leh).
- **Privacy:** only the district name is sent, it is not stored or logged (the access log keeps the path only) and it is not included in AI requests. Results are cached per district for 12 hours instead of a nightly job.
- Live check on 2026-11-10: Mysuru, Pune, Bengaluru, Kolkata, Leh and Chennai all returned weather; an unknown name returns a message.

## Batch: survey and lab-checked screeners (2026-11-10)

All trained by the Curanex team. Rules fixed before testing: logistic unless boosting wins by 0.02; threshold from validation sensitivity 0.80; test scored once; on only with test sensitivity >= 0.80 and AUC >= 0.80.

**US CDC BRFSS** (self-reported diagnosis; trained on 2022, tested on about 430,000 different people in 2023):

| Model | Sens | Spec | AUC | Result |
|---|---:|---:|---:|---|
| Coronary heart disease | 0.816 | 0.710 | 0.839 | On, Verified |
| Stroke | 0.813 | 0.673 | 0.817 | On, Verified |
| COPD | 0.813 | 0.712 | 0.839 | On, Verified |
| Kidney disease | 0.809 | 0.657 | 0.809 | On, Verified |
| Arthritis | 0.820 | 0.637 | 0.807 | On, Verified |
| Depression | 0.797 | 0.706 | 0.830 | Off (sensitivity) |
| Type 2 diabetes | 0.810 | 0.642 | 0.798 | Off (AUC) |
| Cancer, any | 0.823 | 0.571 | 0.760 | Off |
| Asthma | 0.796 | 0.400 | 0.668 | Off |
| High blood pressure (2021 to 2023) | 0.810 | 0.629 | 0.794 | Off (AUC) |
| High cholesterol (2021 to 2023) | 0.807 | 0.487 | 0.705 | Off |

**US CDC NHANES** (condition measured, not reported; trained 2017-18, tested 2021-23 unless noted):

| Model | Sens | Spec | AUC | Result |
|---|---:|---:|---:|---|
| Fatty liver (FibroScan CAP >= 285) from liver tests | 0.826 | 0.710 | 0.850 | On, Verified |
| Iron deficiency (ferritin < 15) from CBC | 0.931 | 0.578 | 0.872 | On, Verified for women and girls 12-49 only (the only group in the test cycle), so offered only to them |
| Liver fibrosis (stiffness >= 8 kPa) | 0.719 | 0.710 | 0.782 | Off |
| Osteoporosis (DXA, 2013-14 to 2017-18) | 0.752 | 0.760 | 0.830 | Off (sensitivity) |
| Kidney disease by eGFR/albuminuria, questions only | 0.853 | 0.534 | 0.790 | Off |
| Prediabetes by HbA1c, questions only | 0.879 | 0.473 | 0.740 | Off |
| Measured high BP, questions only | 0.820 | 0.456 | 0.697 | Off |
| Measured high cholesterol, questions only | 0.765 | 0.424 | 0.644 | Off |

**UCI datasets:**
- Diabetes symptom check (Sylhet): held-out sensitivity 0.812, specificity 0.975, AUC 0.966. On, Unverified (same hospital only).
- Hepatitis C labs: not used, because its healthy group is blood donors (a hand-picked healthy group).
- Cervical cancer risk factors: not used, because without the test results there is no usable signal.

Live check through `/chat`: all 8 new models returned results with their training data, limitations and credit. The form pre-fills "has diabetes" and similar answers from the profile's conditions.

Known issue: the deterministic red-flag rules escalate any message containing "stroke", including "what is my risk of a stroke". The rule was left unchanged; the stroke check is reachable from the attach menu.

## Breast ultrasound, second attempt (2026-11-10)

Trained on BUS-BRA (Brazil) and BUSI (Egypt) together, threshold from BUS-BRA validation (sensitivity 0.90), tested once on BrEaST (Poland, 252 lesions, TCIA): sensitivity 0.898, specificity 0.422, **AUC 0.769**. Fails the bar, so it is not exported and stays off. With frozen ImageNet features and a logistic layer the ceiling on breast ultrasound looks like AUC 0.77-0.79; fine-tuning the network on a GPU is the next thing to try.

NHANES hearing loss (better-ear 500-4000 Hz average > 25 dB, from four hearing questions plus age and sex; trained 2011-12, tested once on 2015-16, 4,308 people): AUC 0.890 but sensitivity 0.769 at the threshold fixed on validation. Fails the sensitivity bar, so it stays off.

## Switch-on bar changed by the user (2026-11-10)

The user decided that a model scoring 0.75 counts as good. Applied as: **live** when the unseen-data test reaches sensitivity >= 0.75 and AUC >= 0.75; **Verified** still needs 0.80 for both, so models between 0.75 and 0.80 are shown as Unverified with their numbers. No model was retrained or re-tuned; already-measured results were re-classified. Switched on under this bar (Unverified): CDC survey depression (sens 0.797, AUC 0.830), type 2 diabetes (0.810, 0.798), cancer of any kind (0.823, 0.760), high blood pressure (0.810, 0.794); NHANES osteoporosis (0.752, 0.830), hidden kidney disease by labs (0.853, 0.790), hearing loss (0.769, 0.890). Still off: asthma, high cholesterol (both), liver fibrosis, prediabetes, measured high BP, colorectal pathology (tumour sensitivity 0.142 on the outside set).

Model search: new descriptions were made specific, and retrieval cases where a sibling model answers the same question equally well list it under `also` (the AI is offered the top 4 models and picks one).

## Cancer batch (2026-11-10)

| Model | Trained on | Outside test (scored once) | Result |
|---|---|---|---|
| Breast ultrasound | BUS-BRA (Brazil) + BUSI (Egypt) | BrEaST, Poland (252): sens 0.898, spec 0.422, AUC 0.769 | **On, Unverified** under the 0.75 bar. Colour check removed, because 42 of 252 real scans carry colour Doppler; the adapter reproduces the test exactly |
| Colorectal pathology tiles | Kather 2016 (Mannheim) | CRC-VAL-HE-7K (NCT, 50 other patients): tumour sens 0.142, AUC 0.852 | Off: misses most tumour tiles at the fixed threshold (stain shift between labs) |
| Cervical Pap smear cells | SIPaKMeD (Greece) | Herlev (Denmark, 917 cells): abnormal sens 0.401, AUC 0.595 | Off |

Cancer coverage now live: skin (melanoma, basal cell carcinoma), brain tumours (MRI), breast lumps (ultrasound), lung nodules (CT), kidney tumours (CT), colon polyps (endoscopy), and a general cancer risk score from the CDC survey. Not feasible on this laptop or with public data that passes an outside test: colorectal and cervical pathology (stain shift), mammography (datasets over 100 GB), and cancers that need 3D scans or biopsies (pancreas, ovary, prostate, liver).

## Batch to 50 models (2026-11-10, evening)

All trained by the Curanex team and tested once on unseen data; live at sensitivity and AUC >= 0.75, Verified at >= 0.80 on an independent set.

| Model | Test | Sens | Spec | AUC | Status |
|---|---|---:|---:|---:|---|
| Heart failure (NHANES) | 2021-23, 7,791 people | 0.913 | 0.703 | 0.909 | Verified |
| Memory and concentration problems (BRFSS) | 2023, 415,195 people | 0.805 | 0.714 | 0.837 | Verified |
| Gout (NHANES) | 2017-18, 5,560 people | 0.855 | 0.596 | 0.784 | Unverified |
| Low bone density, osteopenia (NHANES DXA) | 2017-18, 2,286 people | 0.761 | 0.621 | 0.772 | Unverified |
| Skin cancer, non-melanoma (BRFSS) | 2023 | 0.817 | 0.602 | 0.776 | Unverified |
| Serious vision problems (BRFSS) | 2023 | 0.801 | 0.581 | 0.767 | Unverified |
| Fetal distress from CTG (UCI, clinic) | held-out, 426 | 0.874 | 0.906 | 0.965 | Unverified (same source) |
| Breast FNA cytology (UCI Wisconsin, clinic) | held-out, 93 | 0.833 | 0.978 | 0.991 | Unverified (same source) |
| Mammogram mass (UCI, clinic) | held-out, 192 | 0.820 | 0.806 | 0.894 | Unverified (same source) |
| Glioma grade (UCI TCGA, clinic) | held-out, 168 | 0.873 | 0.856 | 0.932 | Unverified (same source) |
| Liver cirrhosis in PBC (UCI Mayo, clinic) | held-out, 83 | 0.931 | 0.352 | 0.787 | Unverified (same source) |
| Diabetic macular oedema (IDRiD) | official test, 79 | 0.875 | 0.783 | 0.883 | Unverified (same source) |
| Dermoscopy lesion (HAM10000, clinic) | held-out by lesion, 1,494 (scored 2026-11-06, reproduced exactly) | 0.855 | 0.741 | 0.882 | Unverified (same source) |
| Oesophagitis (Kvasir) | held-out, 150 | 0.813 | 0.813 | 0.915 | Unverified (same source) |

Off (below 0.75 on the test): undiagnosed diabetes, high uric acid, thyroid problems, prediabetes (BRFSS), periodontitis, rheumatoid arthritis, liver condition, anaemia, kidney stones, sleep problems, appendicitis (AUC 0.709), psoriasis (sensitivity 0.652), gallstones (sensitivity 0.742), thyroid cancer recurrence (sensitivity 0.667), other retina disease (AUC 0.735).

Live check through `/chat` on 2026-11-10: all 14 newest models returned a result with training data, limitations and credit. The registry now lists 50 live models.

## Verification push: fresh re-tests (2026-11-10)

Rules fixed before any fresh data was scored (`training/verify_brfss_2024.py`, `training/verify_nhanes_fresh.py`): Verified models are scored unchanged on the fresh data as a confirmation. Unverified models use their old test as validation (logistic or boosting, whichever has the higher validation AUC; threshold for validation sensitivity 0.82), then the fresh data is scored once. Verified at >= 0.80 sensitivity and AUC, live at >= 0.75, otherwise off.

| Model | Fresh test | Sens | AUC | Result |
|---|---|---:|---:|---|
| Depression (BRFSS) | 2024, 455,006 people | 0.826 | 0.832 | **Now Verified** |
| Type 2 diabetes (BRFSS) | 2024, 456,636 | 0.817 | 0.805 | **Now Verified** |
| Gout (NHANES) | 2015-16, 5,711 | 0.841 | 0.817 | **Now Verified** |
| Osteoporosis (NHANES DXA) | 2009-10, 2,448 | 0.859 | 0.847 | **Now Verified** |
| Heart disease, stroke, COPD, kidney, arthritis, memory (BRFSS) | 2024, about 455,000 each | 0.812-0.821 | 0.808-0.841 | Verified, confirmed |
| Iron deficiency, heart failure (NHANES) | 2015-16 | 0.894, 0.925 | 0.856, 0.919 | Verified, confirmed |
| General cancer, skin cancer, vision (BRFSS) | 2024 | 0.818-0.823 | 0.768-0.782 | Still live, Unverified |
| Low bone density (NHANES DXA) | 2009-10 | 0.823 | 0.778 | Still live, Unverified |
| Hidden kidney disease by labs (NHANES) | 2015-16, 5,074 | 0.748 | 0.777 | **Switched off** |
| Dermoscopy lesion (HAM10000) | BCN20000, Hospital Clinic Barcelona (ISIC archive), 1,427 lesions | 0.490 | 0.697 | **Switched off** |

Not re-testable yet: fatty liver (no other FibroScan cycle), hearing loss (no adult audiometry in 2009-10), blood pressure (not asked in 2024). HyperKvasir, which could re-test the endoscopy models, downloads at about 90 KB/s (3.9 GB), so it was stopped.

Result: 48 live, 25 Verified (was 50 live, 21 Verified).

## Back to 50, more outside tests (2026-11-10, late)

- **HyperKvasir outside test** (Baerum Hospital, Norway; fetched from OSF; images that copy Kvasir removed by file hash and difference hash; scored once, models unchanged; `tools/evaluate_hyperkvasir.py`):
  - Endoscopy check: colon polyps sensitivity 0.876, specificity 0.988, AUC 0.980 (849 polyps vs 723 normal cecum), passes the Verified bar; ulcerative colitis sensitivity 0.756, AUC 0.985 (694 images). The model stays Unverified as a whole because one of its two findings is below 0.80.
  - Oesophagitis check: sensitivity 0.766, specificity 0.777, AUC 0.848 (462 oesophagitis vs 676 normal Z-line). Stays live, Unverified.
- **Voice health check (Parkinson's)** moved from Experimental to live, Unverified, under the user's 0.75 bar (held-out people: sensitivity 0.886, AUC 0.764). It runs on its own screen (`own_screen`), so it is not offered as a chat tool; the attach menu and Model Library open the screen.
- **Serious hearing difficulty (BRFSS)**, built with the fresh-test procedure from the start (`training/train_brfss_fresh.py`): boosting chosen on 2023 (AUC 0.793), tested once on 2024: sensitivity 0.824, specificity 0.607, AUC 0.792. Live, Unverified.
- Hearing loss (NHANES) could not be re-tested: the 2003-04 cycle asks different hearing questions.

Result: 50 live, 25 Verified, 84 conditions.

## Lung CT full retraining on a Kaggle GPU (2026-11-11)

Notebook `kaggle/lung_ct_finetune.ipynb`, rules fixed in the notebook before running: full fine-tuning of EfficientNet-B0 on IQ-OTH/NCCD (near-duplicates removed, strong augmentation), threshold from validation sensitivity 0.85, outside test once on the Chest CT-Scan collection (1,000 images, 785 cancer). Result: sensitivity 0.284, specificity 0.819, **AUC 0.748**. Fails (below 0.75), so it was not deployed; the live `lung_ct_check` is unchanged and stays Unverified. Cause: IQ-OTH/NCCD is 110 patients from one hospital and one scanner, too narrow to learn what transfers to other scanners.

## Kaggle GPU retraining, round 2 (2026-11-11)

Run directly through the Kaggle CLI (notebooks in `kaggle/run_*`, rules fixed in each notebook before running).

| Model | Design | Outside test (once) | Sens | Spec | AUC | Result |
|---|---|---|---:|---:|---:|---|
| Breast ultrasound | Whole network fine-tuned on BUS-BRA + BUSI | BrEaST, Poland (252) | 0.827 | 0.519 | 0.804 | **Verified, deployed** (replaces the AUC 0.769 model). Through the app's own adapter: 0.827 / 0.539 / 0.808 |
| Lung CT, second design | Trained on the Chest CT-Scan collection | IQ-OTH/NCCD, Iraq (840, malignant vs benign + normal) | 0.050 | 1.000 | 0.825 | Off: ranks well but the threshold does not transfer between hospitals. Live lung CT check unchanged |

Malaria (whole network, stain-colour augmentation, outside test on BBBC041) is queued behind Kaggle's two-GPU limit.

## Verification push, round 3 (2026-11-11)

| Model | New test (once) | Sens | Spec | AUC | Result |
|---|---|---:|---:|---:|---|
| Malaria cell (fully retrained on a Kaggle GPU, stain-colour augmentation) | BBBC041, 1,600 cells | 0.877 | 0.815 | 0.925 | **Verified, deployed** (was AUC 0.661); app adapter reproduces 0.877 / 0.807 / 0.924 |
| Macular oedema | Messidor-2 (France), 1,057 gradable photos, official adjudicated DME | 0.785 | 0.837 | 0.845 | Live, Unverified (sensitivity just under 0.80) |
| Mammogram mass (clinic) | CBIS-DDSM (USA), 781 masses, age and mass density not recorded so typical values used | 0.728 | 0.888 | 0.883 | **Switched off** (sensitivity below 0.75; rule fixed before scoring) |
| Survey: cancer, skin cancer, vision, hearing difficulty, high BP | BRFSS 2025 (third test; retrained on more years, chosen on 2024) | 0.829-0.842 | 0.555-0.605 | 0.766-0.797 | All stay live, Unverified: these questions top out just below AUC 0.80 |

Result: 49 live, 27 Verified.

Kidney CT check: first outside test on the axial kidney stone collection (Kaggle, 2,839 images after removing 525 copies of training scans; stone vs no stone; scored once): sensitivity 0.870, specificity 0.157, **AUC 0.522**. **Switched off.** A swap design (train on the axial collection, test once on CT-KIDNEY) is queued on Kaggle. Endoscopy (train HyperKvasir, test Kvasir v2) and macular oedema (train Messidor-2, test IDRiD) swap designs are running on Kaggle.

Result: 48 live, 27 Verified.

## Swap designs on Kaggle GPUs (2026-11-11)

Train on one collection, test once on another the new model never saw (rules in `kaggle/make_swap_notebooks.py`).

| Model | Train | Outside test (once) | Sens | Spec | AUC | Result |
|---|---|---|---:|---:|---:|---|
| Macular oedema | Messidor-2 (France) | IDRiD (India), all 455 graded photos | 0.875 | 0.827 | 0.943 | **Verified, deployed.** The notebook's copy filter wrongly removed 337 IDRiD photos (fundus images collide at 64-bit hashes; the sets can't share images), leaving 118 (0.886 / 0.833 / 0.924); the full set was scored with the exported model and is the reported result |
| Oesophagitis | HyperKvasir (8 classes) | Kvasir v2, 3,232 copies removed | 0.801 | 0.918 | 0.931 | **Verified, deployed** |
| Ulcerative colitis | same model | Kvasir v2 | 0.883 | 1.000 | 0.999 | **Verified, new check** |
| Colon polyps | same model | Kvasir v2 | 0.405 | 1.000 | 0.999 | Fails; the existing Kvasir model stays for polyps (Verified on HyperKvasir: 0.876 / 0.988 / 0.980) and now reports polyps only |
| Kidney stone | axial stone collection | CT-KIDNEY, 940 copies removed | 0.586 | 0.992 | 0.921 | Off: threshold does not transfer |

Result: 49 live, 31 Verified.
- **2026-11-11, Kaggle GPU round 4/5, full fine-tuning: TB and cervix still not switched on.** TB: EfficientNet-B0 fine-tuned on 8,399 TBX11K X-rays (validation AUC 0.997), then scored once on 800 NLM Shenzhen + Montgomery X-rays: sensitivity 0.746, specificity 0.406, AUC 0.622. Cervical cells: fine-tuned on 4,049 SIPaKMeD cells (validation AUC 0.999), then scored once on 917 Herlev cells: sensitivity 0.101, AUC 0.673. Both still learn the source collection (scanner, stain, crop style), not the disease; full fine-tuning did not fix that. Models kept in kaggle/out_* for the record and not deployed.
- **2026-11-11, Parkinson's voice check, outside test on the Italian Parkinson's Voice and Speech recordings (Kaggle CPU): fails, stays Unverified.** Rules fixed first: live model unchanged, sustained "a" (VA1/VA2), 24 people with Parkinson's vs 22 age-matched elderly healthy controls, one mean score per person, scored once. Result: sensitivity 0.625, specificity 0.091, AUC 0.402 (4 recordings refused). A first run labelled every person as Parkinson's (the dataset's own name contains the word) and was discarded before any control was scored; the model and threshold were not touched. The features do not carry over to a different microphone, room and language.
- **2026-11-11, survey screeners, outside test on NHANES 2017-18 (a different US survey; Kaggle CPU).** Retrained on BRFSS only with the features both surveys share; rules fixed first; scored once. High blood pressure 0.807 / 0.819 and hearing difficulty 0.877 / 0.825 (sensitivity / AUC) pass. Vision loss 0.981 / 0.770 (specificity 0.113), cancer 0.704 / 0.789 and skin cancer 0.667 / 0.826 do not.
- **2026-11-11, colorectal cancer tiles (Kaggle GPU, full fine-tuning with stain augmentation): not switched on.** Trained on Kather 2016 (Mannheim), scored once on CRC-VAL-HE-7K (NCT Heidelberg, 7,180 tiles): AUC 0.977 but sensitivity 0.107 at the threshold fixed on validation. The ranking transfers; the threshold does not (stain shift lowers every score). Under the pre-registered rule this is a fail.
- **2026-11-11, survey screeners: high blood pressure and hearing difficulty now Verified** on NHANES 2017-18 (see above). The boosted trees are stored as plain numbers in the model JSON (no pickle, no ONNX: the converter failed on Kaggle's image) and walked in numpy by the survey adapter; checked against scikit-learn on NHANES rows, maximum difference 0.0. Result: 49 live, 33 Verified.
- **2026-11-11, gestational diabetes, outside test on the Maternal Health and High-Risk Pregnancy dataset (Kaggle, vmohammedraiyyan): fails, stays Unverified.** Model, adapter and "few inputs" threshold unchanged; age, BMI and blood sugar only; pre-existing diabetes left out; scored once (training/verify_gdm_maternal.py): 842 pregnancies (105 GDM), sensitivity 0.962, specificity 0.217, AUC 0.745.
- **2026-11-11, retinal OCT (Kermany 2018, full fine-tuning), outside test on OCTDL (Russia):** AMD vs normal sensitivity 0.904, AUC 0.993 (passes); DME vs normal sensitivity 0.619, AUC 0.979 (fails). The pre-registered rule needed both, so the combined OCT check is not Verified. An AMD-only check is being confirmed once on Duke / Srinivasan 2014 with the model and threshold unchanged.
- **2026-11-11, AMD check from a retinal OCT scan: new, Verified, live (model 50).** The AMD-only check (AMD = CNV + drusen, threshold unchanged) was confirmed once on Duke / Srinivasan 2014 (USA): eye level 30 eyes, sensitivity 1.000, specificity 0.933, AUC 1.000; B-scan level 2,130 scans, sensitivity 0.914, AUC 0.969. Together with OCTDL (0.904 / 0.993) it is Verified. Live check through the torch worker: a Duke AMD scan is flagged, a normal scan is not. Result: 50 live, 34 Verified.
- **2026-11-11, dermoscopy, full fine-tuning on HAM10000 (Kaggle GPU), outside test on 1,500 BCN20000 lesions: fails** (sensitivity 0.601, specificity 0.719, AUC 0.706). Not deployed.
- **2026-11-11, NHANES round 2 (Kaggle CPU), rules fixed first, scored once:** hearing loss retrained on 2011-12 + 2015-16, tested on 2017-18 (ages 70-80): sensitivity 1.000, specificity 0.000, AUC 0.859. It flags everyone at that age, so it is not counted as a pass and not deployed; the old model stays Unverified. Low bone density with fracture, steroid and parent-hip-fracture questions, trained on 2009-10, 2013-14, 2017-18 and tested on 2005-08: sensitivity 0.860, AUC 0.791 (just under). Not swapped in.
- **2026-11-11, survey round 2 (Kaggle CPU): BRFSS 2022-2025 plus education (and white vs not white for skin cancer), tested once on NHANES 2015-16:** skin cancer 0.690 / 0.913, cancer 0.697 / 0.817, vision loss 0.978 / 0.798 (sensitivity / AUC). AUC now passes for two of them, but the threshold chosen on BRFSS does not carry over to NHANES (sensitivity below 0.80). None deployed.
- **2026-11-11, survey round 3: threshold set on one NHANES cycle, tested on another.** Round-2 models unchanged; threshold set once on NHANES 2017-18 (sensitivity >= 0.85 there); scored once on NHANES 2013-14 (never used for these models). **Skin cancer: sensitivity 0.868, specificity 0.753, AUC 0.889 (5,328 adults, 106 cases) - Verified, deployed.** The app now asks two extra questions for this check: family background (White / European, a stand-in for fair skin) and education. Cancer 0.800 / 0.793 and vision loss 0.830 / 0.784 miss on AUC and keep their earlier models (Unverified). Result: 50 live, 35 Verified.
- **2026-11-12, skin-photo check (DermNet model), outside test on Fitzpatrick17k (other atlases; official label file from the authors' GitHub): fails, stays Unverified.** Groups mapped by a fixed list; 12 groups with 30+ photos, 7,566 photos, scored once: average top-3 sensitivity 0.426, average AUC 0.696, first-choice accuracy 0.244. Best group acne and rosacea (AUC 0.866, top-3 0.667); weakest seborrheic keratoses and benign tumours (AUC 0.606). A first run stopped at its own check before scoring because the photo files carry no labels.
- **2026-11-12, replacing the six checks with no outside data (user decision).** Fetal CTG, Wisconsin needle aspirate, PBC cirrhosis, TCGA glioma grade, Sylhet diabetes symptoms and the ILPD liver model have no public dataset from another source with the same inputs, so they can never be Verified. Replacement candidates now running on Kaggle, each with rules fixed first and scored once: hidden diabetes from a routine blood test (NHANES 2017-18 -> 2021-23, no sugar tests as inputs); DME on OCT (threshold set on OCTDL, tested on Duke); colorectal cancer tiles (threshold set on CRC-VAL-HE-7K, tested on NCT-CRC-HE-100K); kidney stone CT (threshold set on CT-KIDNEY, tested on a third Roboflow collection); breast cancer biopsy slides (BreakHis -> BACH).
- **2026-11-12, replacements deployed (Verified, live):** hidden diabetes from a routine blood test (NHANES 2017-18 -> 2021-23, scored once: 5,181 adults, 152 cases, sensitivity 0.954, specificity 0.507, AUC 0.815; no sugar tests as inputs) and DME on a retinal OCT scan (threshold set once on OCTDL; Duke scored once: eye level 30 eyes sensitivity 1.0, AUC 1.0; scan level 2,508 scans sensitivity 0.826, specificity 0.997, AUC 0.971). Live checks: a Duke DME scan is flagged, AMD and normal scans are not; a high-risk report is flagged, a healthy 26-year-old is not. The diabetes symptom check and the glioma grade check are switched off (retired) in their place. Kidney stone CT on the Roboflow set fails (its test had only one stone-free image; sensitivity 0.726). Result: 50 live, 37 Verified; 106 tests pass.
- **2026-11-12, more replacement attempts that failed (rules fixed first, scored once, none deployed):** vision loss round 4 (education + ancestry; threshold set on NHANES 2017-18; tested on 2013-14) 0.844 / 0.779; prediabetes from a routine blood test (NHANES 2021-23) 0.904 / 0.756; brain bleeding on CT (Turkish stroke set -> PhysioNet CT-ICH) slice level 0.053 / 0.690; colorectal tiles (threshold set on CRC-VAL-HE-7K, tested on NCT-CRC-HE-100K, 99,927 tiles) 0.769 / 0.953; pathologic myopia and AMD from a retina photo (ODIR-5K -> RFMiD, thresholds set on RFMiD training, tested on its evaluation + test sets) myopia 0.742 / 0.823, AMD 0.710 / 0.860 (sensitivity / AUC). The breast biopsy run stopped before scoring (the BACH copy used had no labelled photos) and is rerunning on another copy of BACH with the same rules.
- **2026-11-12, NHANES lab round C (rules fixed first; NHANES 2017-18 -> 2021-23, scored once):** low kidney filtering (eGFR < 60, creatinine not an input) 5,443 adults, sensitivity 0.839, specificity 0.846, AUC 0.917 - **Verified, deployed**; hepatitis C (HCV RNA) 5,873 adults but only 27 cases, sensitivity 0.889, specificity 0.724, AUC 0.890 - **Verified, deployed** (small case count noted in the model card); protein in urine 0.885 / 0.754 - not deployed. Breast biopsy slides (BreakHis -> BACH, 400 photos) sensitivity 0.525, AUC 0.687 - fails. The ILPD liver check and the Wisconsin needle-aspirate check are retired in their place. Live checks pass; 106 tests pass. Result: 50 live, 41 Verified.
- **2026-11-12, NHANES lab round D (rules fixed first; 2017-18 -> 2021-23, scored once):** anaemia from MCV, MCH, RDW and platelets (no haemoglobin input) 6,721 people, sensitivity 0.844, specificity 0.565, AUC 0.814 - **Verified, deployed**; fatty liver from questions only (FibroScan truth) 5,525 adults, 0.866 / 0.827 - **Verified, deployed**; high uric acid without uric acid 0.866 / 0.751 - not deployed. Gestational diabetes (failed its outside test) and the PBC cirrhosis check are retired in their place. Live checks pass. Result: 50 live, 43 Verified.
- **2026-11-12, NHANES lab round E: all fail** (sensitivity / AUC): high triglycerides 0.758 / 0.794, low HDL 0.809 / 0.791, high total cholesterol 0.850 / 0.757, raised ALT from questions 0.835 / 0.641.
- **2026-11-12, NHANES rounds F, G, H (rules fixed first; 2017-18 -> 2021-23, scored once; the defining test is never an input).** Verified and deployed: inflammation from a blood count (hs-CRP > 3) 0.842 / 0.807; diabetes from a routine blood test without sugar tests 0.885 / 0.812; metabolic syndrome from questions 0.918 / 0.802; insulin resistance from a routine blood test 0.854 / 0.834; low kidney filtering from questions 0.824 / 0.853; insulin resistance from questions 0.884 / 0.800 (sensitivity / AUC). Failed: measured high BP from labs 0.858 / 0.703; liver stiffness 0.787 / 0.787; low vitamin D 0.858 / 0.699; diabetes or prediabetes from labs 0.891 / 0.790; kidney damage from questions 0.903 / 0.785. Retired in their place: voice check, lung CT, skin condition shortlist, mouth photo check, NHANES hearing loss, survey vision loss (each kept in the registry with the reason). Note on multiple testing: 25 replacement candidates were tried today, each scored once on its own outside set; the passes are reported with their exact numbers so the reader can judge. Result: 50 live, 49 Verified.
- **2026-11-12, NHANES rounds I and J: all fail (scored once):** metabolic syndrome from liver enzymes and uric acid 0.853 / 0.795; inflammation from questions 0.870 / 0.765; hidden diabetes from questions 0.941 / 0.738; airflow obstruction (spirometry FEV1/FVC < 0.70, ages 40-79) from questions, NHANES 2007-10 -> 2011-12, 0.862 / 0.777. The last Unverified live model is the low bone density check (best outside test: sensitivity 0.823, AUC 0.778 on 2009-10; retrained version 0.860 / 0.791 on 2005-08).
- **2026-11-12, NHANES rounds K and L: all fail (rules fixed first, scored once):** peripheral artery disease (ankle-brachial index < 0.90) from questions, 1999-2002 -> 2003-04: 0.824 / 0.765; current depression (PHQ-9 >= 10) from non-PHQ questions, 2017-18 -> 2021-23: 0.588 / 0.742; hepatitis B core antibody from a liver function test and country of birth: 0.831 / 0.754 (sensitivity / AUC). Diabetic foot nerve damage could not be built (the monofilament summary codes left too few labelled people) and was skipped before scoring. The low bone density check stays live and labelled Unverified. Result unchanged: 50 live, 49 Verified.
