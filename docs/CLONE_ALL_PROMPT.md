# Prompt: clone all 126 model repos

Paste everything inside the code block into Claude Code (after Prompt 0, 1 and 2 from CLAUDE_CODE_PROMPTS.md).

```
Clone ALL of the model repositories below into backend/vendor/ and register them, so the AI Doctor can use them.
Read CLAUDE.md, docs/WORKFLOW.md (Part A) and docs/model_registry.json first.

HOW TO CLONE (write this as tools/clone_repos.py, then run it):
1. Check free disk space first and tell me the number. If less than 25 GB is free, stop and ask me.
2. For every URL in the list: shallow clone, without large Git LFS files:
     GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 <url> backend/vendor/<model_id>
   (<model_id> is the second column.) Run up to 6 clones in parallel. Retry once on network errors.
3. After each clone, measure its size. If a repo is larger than 2 GB, delete the dataset/image folders
   inside it (keep code, notebooks, weights, README, LICENSE), and list what you removed. If it is still
   larger than 2 GB, stop and ask me.
4. If a repo's weights are stored with Git LFS (pointer files), fetch ONLY those weight files:
     git -C backend/vendor/<model_id> lfs pull --include="<weight path>"
5. Never run any code from these repos and never pip install their requirements on my machine.
   They are untrusted. They are only read here; models are loaded later inside a Docker
   container with no network (WORKFLOW.md step A4).
6. backend/vendor/ must stay in .gitignore.

FOR EACH CLONED REPO, RECORD (in the registry entry's "audit" field):
- commit SHA, clone date, folder size
- weight files (path, size, SHA-256), framework (sklearn / xgboost / lightgbm / pytorch / tensorflow / ultralytics / onnx)
- licence (from LICENSE file; "none found" if missing). The project owner has permission from every author,
  so a missing licence is NOT a reason to skip a repo. Add "permission": "granted by author" to each entry.
- external weight downloads mentioned in the README (Google Drive / Hugging Face / Zenodo / Kaggle)
- whether it's notebooks only
Do not change "wave", "app_tier", "integrated" or "verified" in this step.

THEN WRITE docs/INTEGRATION_REPORT.md:
- A summary table of all 126: model_id, wave, tier, cloned OK?, size, weights found?, licence, framework.
- For the wave-1 and wave-2 repos (23), a detailed section each: exact input features with units and
  order, preprocessing, output classes, training dataset, which worker (tabular / torch / tf),
  risks, and a verdict: READY / NEEDS WORK (what) / SKIP (why).
- For backlog and hospital-only repos: one line each on why they can't run from the phone app
  (no weights / needs MRI, CT, EEG, ICU data, etc.) and what would make them usable.
- List any repo that failed to clone (deleted, private, renamed) with the error.

Finally report: how many cloned, total disk used, the verdict table for the 23, and any repo you need
my decision on. Stop there. Do not start integrating models until I say so.

REPOSITORIES (URL  model_id), grouped by integration wave:
# wave-1 (8)
https://github.com/AndreyGermanov/yolov8_caries_detector  andreygermanov_yolov8_caries_detector
https://github.com/joystonmenezes/xray-vision  joystonmenezes_xray_vision
https://github.com/mlmed/torchxrayvision  mlmed_torchxrayvision
https://github.com/Moizah1/Diabetes-Prediction  moizah1_diabetes_prediction
https://github.com/OmerKurtulus/skin-disease-classification  omerkurtulus_skin_disease_classification
https://github.com/singh2010nidhi/Heart-Failure-Prediction-using-MSAzure  singh2010nidhi_heart_failure_prediction_using_msazure
https://github.com/Sneha798/Heart-Disease-Prediction-Project  sneha798_heart_disease_prediction_project
https://github.com/yahyazuher/AI-Liver-Disease-Diagnosis-System  yahyazuher_ai_liver_disease_diagnosis_system
# wave-2 (15)
https://github.com/Ceren-Kaya/boneage_estimation  ceren_kaya_boneage_estimation
https://github.com/cilab-ufersa/thyroid_disease_AI  cilab_ufersa_thyroid_disease_ai
https://github.com/Diabetic-Research-Group/diabetes-prediction  diabetic_research_group_diabetes_prediction
https://github.com/Ghifar-Khder/knee-osteoarthritis-classification  ghifar_khder_knee_osteoarthritis_classification
https://github.com/ishad05/anemia_ml_project  ishad05_anemia_ml_project
https://github.com/khaled-ghanem/Ocular-Disease-Classification  khaled_ghanem_ocular_disease_classification
https://github.com/madhukalita/Health-Information-and-Diagnostic-System  madhukalita_health_information_and_diagnostic_system
https://github.com/niaid/tb_lesion_cxr_segmentation  niaid_tb_lesion_cxr_segmentation
https://github.com/rash200319/two_stage_preeclampsia_AI_model  rash200319_two_stage_preeclampsia_ai_model
https://github.com/SanyaB1801/Thyroid-Disease-Prediction  sanyab1801_thyroid_disease_prediction
https://github.com/SathwikaEga/pcos-detection  sathwikaega_pcos_detection
https://github.com/seunghahh/PredictingHypertension  seunghahh_predictinghypertension
https://github.com/Sina-Imaginative/dental-caries-detection  sina_imaginative_dental_caries_detection
https://github.com/srishti-chordia/Gestational-Diabetes-Prediction-using-DL  srishti_chordia_gestational_diabetes_prediction_using_dl
https://github.com/yudaleng/COPD-Early-Prediction  yudaleng_copd_early_prediction
# backlog (34)
https://github.com/anchor-neuro/retguard  anchor_neuro_retguard
https://github.com/Bhuvanaa28/Sentiment-Analysis-for-Depression-Detection  bhuvanaa28_sentiment_analysis_for_depression_detection
https://github.com/CASALab-Unisa/Dino2-DR  casalab_unisa_dino2_dr
https://github.com/darwinai/TuberculosisNet  darwinai_tuberculosisnet
https://github.com/deettoh/skin-lesion-classification  deettoh_skin_lesion_classification
https://github.com/djparente/uti-ml  djparente_uti_ml
https://github.com/Duggineniakhil/oral-cancer-detection  duggineniakhil_oral_cancer_detection
https://github.com/Eeman1113/dental_study  eeman1113_dental_study
https://github.com/Holy-Mahdi/ibids-xray-fracture-detection  holy_mahdi_ibids_xray_fracture_detection
https://github.com/ifyoungnet/MedGDM  ifyoungnet_medgdm
https://github.com/lanagarmire/PE_delivery  lanagarmire_pe_delivery
https://github.com/li-xirong/mmc-amd  li_xirong_mmc_amd
https://github.com/m-aljasem/pediatric-radiology-ai  m_aljasem_pediatric_radiology_ai
https://github.com/MarcoParola/detection_framework  marcoparola_detection_framework
https://github.com/MarcoParola/oral_segmentation  marcoparola_oral_segmentation
https://github.com/nawaz127/A-Multilayer-Classification-Framework-for-Skin-Disease-  nawaz127_a_multilayer_classification_framework_for_skin_disease
https://github.com/nyukat/breast_cancer_classifier  nyukat_breast_cancer_classifier
https://github.com/nyukat/gmic  nyukat_gmic
https://github.com/PingjunChen/GradingKneeOA  pingjunchen_gradingkneeoa
https://github.com/pradamehavannan/OCT-Based-Retinal-Disease-Classification-and-AI-Clinical-Decision-Support  pradamehavannan_oct_based_retinal_disease_classification_and_ai_clinical_decision_support
https://github.com/PRAISE-Lab-Repository/PEPrML  praise_lab_repository_peprml
https://github.com/resplab/accept  resplab_accept
https://github.com/rick0110/cervical_cancer  rick0110_cervical_cancer
https://github.com/SakH1075/Multiclass-Prediction-of-Thalassemia-Disorders-Using-Machine-Learning-Models-on-CBC-Parameters  sakh1075_multiclass_prediction_of_thalassemia_disorders_using_machine_learning_models_on_cbc_parameters
https://github.com/sathvika-09/TB-detection  sathvika_09_tb_detection
https://github.com/sfu-mial/skin-IAV  sfu_mial_skin_iav
https://github.com/ShaonINT/Glaucoma_Detection  shaonint_glaucoma_detection
https://github.com/sibisai/skin-scan  sibisai_skin_scan
https://github.com/smon2006/Skin-Disease-Detection  smon2006_skin_disease_detection
https://github.com/uxlabspk/NeuroVision  uxlabspk_neurovision
https://github.com/Vankudoth-Sunil-Kumar/DiseasePrediction  vankudoth_sunil_kumar_diseaseprediction
https://github.com/vineet416/Chronic-Kidney-Disease-Prediction  vineet416_chronic_kidney_disease_prediction
https://github.com/Whitedevil71/FundusNext  whitedevil71_fundusnext
https://github.com/yehonatanke/Thyroid-Predict-ML  yehonatanke_thyroid_predict_ml
# hospital-only (69)
https://github.com/aabo5/Malaria-Detection  aabo5_malaria_detection
https://github.com/AArjcode/UTI_MachineLearning_HAI_vs_CAI  aarjcode_uti_machinelearning_hai_vs_cai
https://github.com/AmirAynede/LC25000-Cancer-Classification  amiraynede_lc25000_cancer_classification
https://github.com/aramis-lab/AD-DL  aramis_lab_ad_dl
https://github.com/arinova2701/SickleShield  arinova2701_sickleshield
https://github.com/bbleier/multimodal-icu-mortality  bbleier_multimodal_icu_mortality
https://github.com/BCV-Uniandes/Cardium  bcv_uniandes_cardium
https://github.com/BIMCV-CSUSP/BIMCV-Prostate-Classification  bimcv_csusp_bimcv_prostate_classification
https://github.com/BioMedIA-MBZUAI/FetalCLIP  biomedia_mbzuai_fetalclip
https://github.com/BlackCatDisha/EpiSentinel  blackcatdisha_episentinel
https://github.com/BoranSekeroglu/OSTEO_MODELS  boransekeroglu_osteo_models
https://github.com/conze/SlicerPolycysticKidneySeg  conze_slicerpolycystickidneyseg
https://github.com/czifan/Multimodal-Medicine-AI  czifan_multimodal_medicine_ai
https://github.com/darshan-cpp/NeuroML-PTSD  darshan_cpp_neuroml_ptsd
https://github.com/Darshan0109/all-leukemia-detection  darshan0109_all_leukemia_detection
https://github.com/dayu-lab/DB-EpiFM  dayu_lab_db_epifm
https://github.com/DIAGNijmegen/PANORAMA_baseline  diagnijmegen_panorama_baseline
https://github.com/DREAMXFAR/Pos4CHD  dreamxfar_pos4chd
https://github.com/EranZvuloni/MSD-generalization  eranzvuloni_msd_generalization
https://github.com/ESandML/FCBFormer  esandml_fcbformer
https://github.com/ezequieldlrosa/DeepIsles  ezequieldlrosa_deepisles
https://github.com/fredyandres73/Pressure-Injury-ML-prediction-code  fredyandres73_pressure_injury_ml_prediction_code
https://github.com/GatorAIM/BAT-AKI  gatoraim_bat_aki
https://github.com/glydi/i2ireg-clfnet  glydi_i2ireg_clfnet
https://github.com/GwadeSteve/MTTL-Research-Malaria  gwadesteve_mttl_research_malaria
https://github.com/harish-kush/OCD-CLASSIFIER  harish_kush_ocd_classifier
https://github.com/hjiangaz/Smart-CCS  hjiangaz_smart_ccs
https://github.com/horus-bot/Swasthya_Predective-system  horus_bot_swasthya_predective_system
https://github.com/hunain6/Dengue-Risk-Prediction-System  hunain6_dengue_risk_prediction_system
https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast  ivadomed_seg_sc_ms_lesion_multicontrast
https://github.com/Jab7/lung-cancer-classification-xai  jab7_lung_cancer_classification_xai
https://github.com/janaristevska/eeg-adhd-detection2  janaristevska_eeg_adhd_detection2
https://github.com/kasikrit/IDA-THL-Classification  kasikrit_ida_thl_classification
https://github.com/keerthikkn/Hospital_readmission_prediction  keerthikkn_hospital_readmission_prediction
https://github.com/LNS98/deep-learning-medical-app  lns98_deep_learning_medical_app
https://github.com/lst627/CARDIAC-FM  lst627_cardiac_fm
https://github.com/Mahmoud-Elbattah/glenda-xai  mahmoud_elbattah_glenda_xai
https://github.com/maynord/7T-MS-lesion-segmentation  maynord_7t_ms_lesion_segmentation
https://github.com/mazder1/ECG  mazder1_ecg
https://github.com/Mesolongo/breast-cancer-classifier-comparison  mesolongo_breast_cancer_classifier_comparison
https://github.com/MikhailYadgarov/Sepsis-prediction  mikhailyadgarov_sepsis_prediction
https://github.com/mini-project-sleepap/sleep-apnea-screening-tool  mini_project_sleepap_sleep_apnea_screening_tool
https://github.com/mmuratarat/kvasir-v2-ViT-classifier  mmuratarat_kvasir_v2_vit_classifier
https://github.com/mnaeimaei/parkinson-datscan-classifier  mnaeimaei_parkinson_datscan_classifier
https://github.com/mohnabilfadl/Parkinson_Detection_Using_Voice_Analysis  mohnabilfadl_parkinson_detection_using_voice_analysis
https://github.com/MrGiovanni/R-Super  mrgiovanni_r_super
https://github.com/mtichavsky/eeg  mtichavsky_eeg
https://github.com/Muhammad-Hassan12/XPathology-Colorectal-Tissue-Specialist  muhammad_hassan12_xpathology_colorectal_tissue_specialist
https://github.com/P-MCC/Leukemia-Detection  p_mcc_leukemia_detection
https://github.com/pazjin/FibUCL  pazjin_fibucl
https://github.com/PGC-PTSD-EWAS/PTSD-Methylation-Risk-Score-2024  pgc_ptsd_ewas_ptsd_methylation_risk_score_2024
https://github.com/pipi0616/hears-lm  pipi0616_hears_lm
https://github.com/Project-MONAI/model-zoo  project_monai_model_zoo
https://github.com/r1sephoenix/ocd_classification  r1sephoenix_ocd_classification
https://github.com/rajpurkarlab/panendosuite  rajpurkarlab_panendosuite
https://github.com/ronggnahc2026/SLOR  ronggnahc2026_slor
https://github.com/rvandewater/YAIB-models  rvandewater_yaib_models
https://github.com/sdoerrich97/true-colon  sdoerrich97_true_colon
https://github.com/seantangth/cvpr2026-cv4chl  seantangth_cvpr2026_cv4chl
https://github.com/Seherin-Zaman/Brain-Tumor-MRI-Dataset  seherin_zaman_brain_tumor_mri_dataset
https://github.com/ShallenCrissle/EndoInsights  shallencrissle_endoinsights
https://github.com/SOCR/PressureInjuryPrediction  socr_pressureinjuryprediction
https://github.com/Subangkar/Simsig  subangkar_simsig
https://github.com/Tabrisrei/ISLES22_SEALS  tabrisrei_isles22_seals
https://github.com/tadeephuy/CoFo  tadeephuy_cofo
https://github.com/uthmri/Deep-Learning-MS-Segmentation  uthmri_deep_learning_ms_segmentation
https://github.com/vaibhav-k/Dengue-Prediction-ML  vaibhav_k_dengue_prediction_ml
https://github.com/Vraj-Data-Scientist/prediction-on-hospital-readmission  vraj_data_scientist_prediction_on_hospital_readmission
https://github.com/yangzhou12/OPENEye-FM  yangzhou12_openeye_fm
```
