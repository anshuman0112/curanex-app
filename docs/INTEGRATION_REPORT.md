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
