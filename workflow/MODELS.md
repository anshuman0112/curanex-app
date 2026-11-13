# Live models (generated 2026-11-14 from backend/registry/model_registry.json)

Regenerate: see workflow/ENVIRONMENT.md -> 'Registry'. Status: Verified = passed an outside test (sensitivity and AUC >= 0.80).

**50 live, 50 Verified.**

| # | Status | Model | id | Outside test (sens / AUC) |
|---|---|---|---|---|
| 1 | Verified | AMD check (retinal OCT scan) | `amd_oct_check` | 0.904 / 0.993 |
| 2 | Verified | Anaemia check (blood count without haemoglobin) | `nhanes_anemia_labs` | 0.844 / 0.814 |
| 3 | Verified | Arm and hand X-ray check | `joystonmenezes_xray_vision` | 0.836 / 0.903 |
| 4 | Verified | Arthritis check | `survey_arthritis` | 0.821 / 0.81 |
| 5 | Verified | Brain MRI check | `brain_mri_check` | 0.871 / 0.935 |
| 6 | Verified | Breast ultrasound check | `breast_ultrasound_check` | 0.827 / 0.804 |
| 7 | Verified | Cancer (any, except skin) check | `survey_cancer` | 0.86 / 0.804 |
| 8 | Verified | Cataract check (retina photo) | `khaled_ghanem_ocular_disease_classification` | 0.893 / 0.939 |
| 9 | Verified | Chronic kidney disease check | `survey_kidney_disease` | 0.814 / 0.808 |
| 10 | Verified | Colon polyp check (endoscopy) | `endoscopy_check` | 0.876 / 0.98 |
| 11 | Verified | COPD (chronic lung disease) check | `survey_copd` | 0.817 / 0.839 |
| 12 | Verified | Coronary heart disease or heart attack check | `survey_heart_disease` | 0.812 / 0.836 |
| 13 | Verified | Depression check | `survey_depression` | 0.826 / 0.832 |
| 14 | Verified | Diabetes check (routine blood test, no sugar test) | `nhanes_diabetes_any_labs` | 0.885 / 0.812 |
| 15 | Verified | Diabetic retinopathy check | `diabetic_retinopathy_check` | 0.966 / 0.943 |
| 16 | Verified | DME check (retinal OCT scan) | `dme_oct_check` | 0.826 / 0.971 |
| 17 | Verified | Fatty liver check (blood tests) | `nhanes_fatty_liver` | 0.826 / 0.85 |
| 18 | Verified | Fatty liver check (no blood test needed) | `nhanes_fatty_liver_questions` | 0.866 / 0.827 |
| 19 | Verified | Glaucoma check (retina photo) | `pamixsun_glaucoma` | 0.907 / 0.869 |
| 20 | Verified | Gout risk check | `nhanes_gout` | 0.841 / 0.817 |
| 21 | Verified | Heart failure risk check | `nhanes_heart_failure` | 0.925 / 0.919 |
| 22 | Verified | Heart muscle injury risk (troponin) | `nhanes_heart_damage_troponin` | 0.856 / 0.905 |
| 23 | Verified | Heart strain check (NT-proBNP risk) | `nhanes_heart_strain_ntprobnp` | 0.868 / 0.878 |
| 24 | Verified | Hepatitis C check (liver function test) | `nhanes_hepatitis_c_labs` | 0.889 / 0.89 |
| 25 | Verified | Hidden diabetes check (routine blood test) | `nhanes_hidden_diabetes_labs` | 0.954 / 0.815 |
| 26 | Verified | High blood pressure check | `survey_high_blood_pressure` | 0.807 / 0.819 |
| 27 | Verified | Inflammation check (blood count and albumin) | `nhanes_inflammation_crp_labs` | 0.842 / 0.807 |
| 28 | Verified | Insulin resistance check (routine blood test) | `nhanes_insulin_resistance_labs` | 0.854 / 0.834 |
| 29 | Verified | Insulin resistance risk (no blood test needed) | `nhanes_insulin_resistance_questions` | 0.884 / 0.8 |
| 30 | Verified | Iron deficiency check (blood count) | `nhanes_iron_deficiency` | 0.894 / 0.856 |
| 31 | Verified | Kidney damage check (routine blood test) | `nhanes_kidney_damage_labs` | 0.845 / 0.8 |
| 32 | Verified | Kidney disease check (hospital report, Indian data) | `ckd_india_report_check` | 0.914 / 1.0 |
| 33 | Verified | Kidney function check (urea and uric acid) | `nhanes_low_egfr_labs` | 0.839 / 0.917 |
| 34 | Verified | Kidney function risk (no blood test needed) | `nhanes_low_egfr_questions` | 0.824 / 0.853 |
| 35 | Verified | Knee X-ray arthritis check | `ghifar_khder_knee_osteoarthritis_classification` | 0.809 / 0.965 |
| 36 | Verified | Low muscle mass check (age 20-59) | `nhanes_low_muscle_mass` | 0.819 / 0.942 |
| 37 | Verified | Macular oedema check (retina photo) | `macular_edema_check` | 0.875 / 0.943 |
| 38 | Verified | Malaria blood cell check | `malaria_cell_check` | 0.877 / 0.925 |
| 39 | Verified | Memory and concentration problems check | `survey_memory_problems` | 0.818 / 0.841 |
| 40 | Verified | Metabolic syndrome check (no blood test needed) | `nhanes_metabolic_syndrome_questions` | 0.918 / 0.802 |
| 41 | Verified | Muscle weakness check (age 50+) | `nhanes_low_grip_questions` | 0.821 / 0.831 |
| 42 | Verified | Oesophagitis check (endoscopy) | `esophagitis_check` | 0.801 / 0.931 |
| 43 | Verified | Osteoporosis risk check | `nhanes_osteoporosis` | 0.859 / 0.847 |
| 44 | Verified | Serious hearing difficulty check | `survey_hearing_difficulty` | 0.877 / 0.825 |
| 45 | Verified | Skin cancer (not melanoma) check | `survey_skin_cancer` | 0.868 / 0.889 |
| 46 | Verified | Skin lesion check | `skin_lesion_check` | 0.88 / 0.943 |
| 47 | Verified | Stroke check | `survey_stroke` | 0.814 / 0.815 |
| 48 | Verified | Teeth photo check | `andreygermanov_yolov8_caries_detector` | 0.894 / 0.929 |
| 49 | Verified | Type 2 diabetes check | `survey_diabetes` | 0.817 / 0.805 |
| 50 | Verified | Ulcerative colitis check (endoscopy) | `ulcerative_colitis_check` | 0.883 / 0.999 |

## Retired (switched off, kept in the registry with the reason)

| id | Why |
|---|---|
| `moizah1_diabetes_prediction` | Failed its outside test (Frankfurt, new patients only: AUC 0.786); replaced by the muscle weakness check, Verified on NHANES 2013-14. Diabetes stays covered by four Verified checks. |
| `srishti_chordia_gestational_diabetes_prediction_using_dl` | Failed its outside test (maternal health dataset, AUC 0.745); replaced by the anaemia check (blood count), Verified on NHANES 2021-23. |
| `yahyazuher_ai_liver_disease_diagnosis_system` | Replaced by the hepatitis C check (liver function test), Verified on NHANES 2021-23; no outside data with the ILPD inputs and labels passed. |
| `skin_condition_check` | Failed its outside test (Fitzpatrick17k: average AUC 0.696); replaced by the metabolic syndrome check, Verified on NHANES 2021-23. |
| `oral_condition_check` | No outside mouth-photo set with the same labels exists; replaced by the insulin resistance check, Verified on NHANES 2021-23. |
| `lung_ct_check` | Failed outside its hospital and no third lung CT set exists; replaced by the diabetes check (routine blood test), Verified on NHANES 2021-23. |
| `voice_parkinsons_check` | Failed its outside test (Italian recordings, AUC 0.402); replaced by the inflammation check (hs-CRP), Verified on NHANES 2021-23. The voice screen stays as an Unverified demo. |
| `diabetes_symptom_check` | Replaced by the hidden diabetes check (routine blood test), Verified on NHANES 2021-23. |
| `clinic_fetal_ctg` | Replaced by the kidney damage check (routine blood test), Verified on NHANES 2021-23; no outside CTG data with the same inputs exists. |
| `clinic_breast_fna` | Replaced by the kidney function check (urea and uric acid), Verified on NHANES 2021-23; no outside cytology data with the same inputs exists. |
| `clinic_cirrhosis` | Replaced by the fatty liver check from questions (FibroScan truth), Verified on NHANES 2021-23; no outside PBC data with the same inputs exists. |
| `clinic_glioma_grade` | Replaced by the DME check on OCT, Verified on Duke; no outside data exists with the same gene-test inputs. |
| `nhanes_low_bone_density` | Missed the 0.80 bar on every outside test (best AUC 0.791); replaced by the kidney disease check trained in India and Verified in Bangladesh. The Verified osteoporosis check covers bone health. |
| `nhanes_hearing_loss` | Missed the 0.80 bar on its outside tests (sensitivity 0.769); replaced by the kidney function risk check (questions), Verified on NHANES 2021-23. The Verified hearing difficulty check covers hearing. |
| `survey_vision_loss` | Missed the 0.80 bar on four outside tests (best AUC 0.798); replaced by the insulin resistance risk check (questions), Verified on NHANES 2021-23. |
| `sneha798_heart_disease_prediction_project` | Failed outside tests (Long Beach VA AUC 0.750, Switzerland 0.784; Iran Z-Alizadeh Sani sensitivity 0.708); replaced by the heart strain check (NT-proBNP), Verified on NHANES 2003-04. |
| `cilab_ufersa_thyroid_disease_ai` | Failed its outside test (NHANES 2007-12, truth low free T4: AUC 0.594); replaced by the low muscle mass check (DXA), Verified on NHANES 2017-18. |
| `mlmed_torchxrayvision` | Failed outside tests (NLM AUC 0.727; TBX11K sensitivity 0.798) and no clean adult test set is left; replaced by the heart muscle injury check (troponin), Verified on NHANES 2003-04. |
