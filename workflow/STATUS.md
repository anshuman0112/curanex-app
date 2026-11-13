# Curanex status (last updated 2026-11-12)

Legend: **DONE** · **PENDING** · **WAITING ON USER**

Current numbers: **50 live models, 47 Verified** (Unverified: heart disease, underactive thyroid, chest X-ray "abnormal" layer).
Git: branch `master`, local commits ahead of `origin/master` (not pushed; push only on the owner's "Yes, push").

## Phase 1-2: platform (backend, workers, model router)
- DONE - FastAPI API (`backend/api`), workers tabular / torch (Docker) / tf (Docker), registry + embeddings router, orchestrator tool loop.
- DONE - Safety: red-flag rules, dose stripping, "not a diagnosis" wording, audit hashing.

## Phase 3-7: models
- DONE - 126 repos audited; wave-1/2 integrated or replaced; Curanex-trained replacements.
- DONE - Outside-test programme (2026-11-11/12): ~45 candidates tried on Kaggle, results in `docs/INTEGRATION_REPORT.md`.
- DONE - 50 live models (see `MODELS.md`).
- DONE - Outside tests of older "Verified" models: knee X-ray passes (India); Pima diabetes fails (retired, replaced by muscle weakness); heart, thyroid, chest X-ray now Unverified.
- DONE (fail) - 2026-11-13 second outside tests: heart on Z-Alizadeh Iran (AUC 0.832, sens 0.708), thyroid on NHANES low free T4 (AUC 0.594), chest X-ray no clean set left. All 3 stay live, Unverified. WAITING ON USER: keep as Unverified, or switch off/replace.
- PENDING - Outside tests for 3 models still tested only on their own source: teeth photo (DentalAI), arm and hand X-ray (MURA), skin lesion (PAD-UFES-20).

## Phase 8: extra routes
- DONE - Route C voice health check (Unverified demo screen; model failed the Italian outside test and is retired from routing).
- DONE - Route D area alerts (weather, experimental). Route A DICOM upload with tag stripping.
- PENDING - Route E clinic portal -> becomes the doctor portal (see Phase 10).
- PENDING - Route B wearables (dropped for now; maybe a Noise smartwatch later).

## Phase 9: final verification
- PENDING - Full live run of every model, red-team set, screenshots at 390 px and 1366 px, latency and memory, update `docs/VERIFICATION.md`.

## Phase 9b: patient app polish (2026-11-12)
- DONE - Traffic-light flag on every model result: green "No problems found in these results" / amber "Please see a doctor"; red stays the emergency rules. (`result_flag` in `backend/api/orchestrator/tools.py`, card in `src/components/blocks.tsx`, test `backend/tests/test_result_flag.py`)
- DONE - Quick-relief medicines for minor problems from the fixed OTC list, no doses, blocked for children/pregnancy/clashing conditions (`backend/api/orchestrator/care.py`).
- PENDING - Fill model forms by voice (patient says the values, AI fills the form, reads it back for a yes).
- PENDING - "Share with a doctor" button (needs Phase 10 storage and roles).

## Phase 10: doctor platform (next phase) - see `DOCTOR_PLATFORM_PLAN.md` and the approved plan below
- DONE (2026-11-12) - Test fixes after model changes: report reader offers Verified checks only (TG/HDL -> diabetes labs, BUN/uric acid -> kidney function); retriever `exclude_if` words (pregnancy, rash, jaundice) stop wrong routing; retrieval cases updated; 109 tests pass.
- DECIDED (2026-11-12) - The 80 books could not be found: use a free medical library instead (StatPearls, ICMR STWs, WHO, DailyMed/openFDA drug labels, MedlinePlus). The owner may add some books later through the same pipeline (local only).
- DECIDED - No-hallucination design: AI proposes diseases, library verifies (supported list with citations; unsupported shown as 'also consider', no treatment); every sentence checked against its cited chunk; doses copied only from drug labels.
- RULE (owner) - The doctor module is built in new folders only; it must not change the live models, registry, AI doctor or patient screens.
- DONE (2026-11-12, 71fda9b) - Accounts and roles (SQLite `backend/data/curanex.db`, scrypt passwords, 7-day tokens): patient / doctor (pending -> approved by admin by hand at /admin) / admin (ADMIN_EMAIL). Server-side role checks.
- DONE (2026-11-12) - Health record (SQLite, Fernet-encrypted, audit log): patient cases, share with a doctor picked from a list of approved doctors (no emails shown), stop sharing, doctor confirms a plan, patient sees 'Your doctor has advised' at /record.
- PENDING - Report upload reading (photo/PDF -> values, patient confirms).
- DONE (2026-11-12, bb57373) - Free library built: `backend/knowledge/` fetchers (StatPearls 91,517 passages, MedlinePlus 1,180, FDA drug labels 1,361 for 113 medicines; exact-ingredient match), `ingest.py` -> `backend/data/knowledge/library.db` (SQLite + FTS5; git-ignored). ICMR STWs: licence forbids copying text -> link to official PDFs only (PENDING).
- DONE (2026-11-12) - Library vectors made on Kaggle GPU (`kaggle/run_library_embed`, 94,058 x 384, ~5 min) -> `backend/data/knowledge/embeddings.npy` + `embedding_ids.json`; hybrid search verified on pneumonia/TB/cough queries.
- DONE (2026-11-12) - First live end-to-end run (owner's example: 45 M, fever 3 days, cough, breathlessness): tests with citations; differential with 7 supported pneumonia-type diagnoses + PE in 'also consider'; treatment: beta-lactam + macrolide (cited), ceftriaxone and azithromycin with label dosing. Fixes from the run: Groq 429 back-off (retry-after, 6 tries), evidence must mention the patient's findings, Unicode hyphen/Greek normalisation in the checker.
- PENDING - 7 drug labels missing (openFDA rate limit): psyllium, atorvastatin, rosuvastatin, aspirin, clopidogrel, sofosbuvir+velpatasvir, fluticasone -> rerun `python -m knowledge.fetch_druglabels` then `python -m knowledge.ingest`.
- DONE (2026-11-12, 63dae74) - Doctor RAG backend: `api/knowledge/` search (hybrid), answer (tests / differential / treatment), verify (statement-level evidence check), drugs (label-only doses, allergy vs contraindications), own temperature-0 LLM client; `/doctor/tests|differential|treatment|medicine` locked by `DOCTOR_DEV_KEY` header until roles exist. Tests: `test_doctor_api.py`, `test_knowledge_verify.py` (119 total pass).
- DONE (2026-11-12) - Eval set `backend/tests/knowledge_cases.json` (60 questions): recall@1 0.600, recall@5 0.867.
- DONE (2026-11-13) - 5-case live run `python -m knowledge.run_cases` -> `backend/data/knowledge/five_cases.json` (pneumonia, type 2 diabetes, TB, iron-deficiency anaemia, CKD). Run 1: right disease in the supported top 4 in 5/5 but top-1 only 2/5 (list was sorted by passage count) and 3/5 treatment plans empty. Fixes: keep the AI's likelihood order; a medicine is kept if named in its statement OR its cited passage (was silently dropping isoniazid-type answers); 'X is contraindicated' becomes a Caution, never a medicine card (metformin at eGFR 27). Run 3: top-1 correct 5/5; plans for all 5; meds: metformin (T2DM), rifampin/ethambutol/pyrazinamide (TB), ACE inhibitor/ARB + metformin caution (CKD); pneumonia and anaemia plans have no medicine card (library statements did not name one).
- DONE - Timing (Groq free tier, empty cache): tests 21-29 s, differential 93-117 s, treatment 21-24 s per case. Same case again (answer cache): ~1-12 s per stage (55/55 cache hits). Groq returns cached_tokens null -> no provider prompt caching; our SQLite answer cache (30 days) is what makes repeats fast.
- PENDING - Manual faithfulness audit of ~30 displayed statements against their cited passages (automatic checker runs on every statement already).
- DONE (2026-11-12) - Doctor workspace screen `src/app/doctor-portal.tsx` (+ `src/lib/doctorApi.ts`): patient details, symptoms -> tests, report -> likely diseases (+ also consider), treatment with label doses (expandable), sources as links; loading/empty/error states. Checked live in the browser with the owner's example. Doctor key lives in `backend/.env` as DOCTOR_DEV_KEY (git-ignored).
- DONE (2026-11-13) - Doctor Desk (owner, 2026-11-13): separate portal at /doctor-portal behind a PIN (the PIN is DOCTOR_DEV_KEY in the git-ignored backend/.env; 10 wrong PINs lock PIN entry 10 min). Home screen card 'Doctor Desk' opens it; patient profile has no doctor/admin links. Doctor login inside is optional (only for shared cases + sending a plan).
- DONE (2026-11-13) - 'Ask the AI about this patient' chat in Doctor Desk (`/doctor/ask`): library answer lines with sources (each checked), plus at most 3 'General AI knowledge (not found in the library)' points; no AI doses; last 6 turns as context; adult cases never cite paediatric chapters. Live: 6-19 s per question.
- NOTE (2026-11-13) - Image checks need Docker Desktop running + `docker start backend-worker-torch-1 backend-worker-tf-1`. A pneumonia X-ray uploaded as 'Medical report' fails (that reads text); use 'Chest X-ray' (tested: Strep pneumonia CXR -> abnormal 1.00, opacity 0.96, pneumonia 0.80, amber). The report error now says so.
- PENDING - Evaluation: 100 test questions, retrieval recall and citation faithfulness.

## Phase 11: upload anything (2026-11-13)
- DONE - One "Upload anything" button in the AI Doctor (`/v1/identify`, `backend/api/vision.py`): the free Groq vision model (`qwen/qwen3.8-27b`) only recognises the file; lab reports/PDFs -> report reader; chest/knee/arm-hand X-ray, brain MRI, skin, fundus, OCT, breast ultrasound, endoscopy, blood smear -> the matching live model; ECG, CT, spine X-ray etc. (no model) -> AI description labelled "AI reading, not a diagnosis". Tested on 10 real/synthetic files: all recognised and routed correctly (3 s each; ~20 s when Groq rate-limits bursts). 140 tests pass.
- DONE - Image workers now restart with Docker (`docker update --restart unless-stopped`). Docker Desktop itself must still be started after a PC restart.
- NOTE - Brain MRI sample (glioma) was called pituitary tumour 0.99: tumour yes, type wrong. Model accuracy, not routing.
- DONE (2026-11-13) - Retest of the 3 Unverified models:
  - Chest X-ray (mlmed_torchxrayvision): TBX11K (2,397 X-rays, China, never in training; healthy vs sick/TB). Abnormal check AUC 0.865. Rule fixed first: flagging point chosen on NLM X-rays for 85% sensitivity -> TBX11K sensitivity 0.798, specificity 0.750 (bar 0.80) -> stays Unverified (misses by 0.002). Live flagging point (set on VinBig) gives 0.688 / 0.911 on TBX11K. Option for the owner: switch the live flagging point to the NLM one (more X-rays flagged, more false alarms).
  - Heart (sneha798): needs exercise-test/thal/ca values that exist only in the 4 UCI hospital files it was already tested on -> no new outside data -> stays Unverified.
  - Thyroid (cilab): live model is effectively TSH-based; outside data (NHANES) defines hypothyroidism by TSH too, so a test would be circular -> stays Unverified.
- DONE (2026-11-13, owner chose it) - Chest X-ray abnormal check now flags at the NLM-chosen point (threshold 0.2172, old VinBig point kept as `threshold_vinbig` in `abnormal_adult_head.json`): outside TBX11K sensitivity 0.798, specificity 0.750 (was 0.688 / 0.911). Limitations text updated; torch worker image re-committed. Still Unverified.
- DONE (2026-11-13, owner) - Upload anything: if no model fits OR the file type is unclear, it goes straight to the AI with the image reader's description, labelled "AI reading".

## Phase 12: per-user cloud storage, doctor-style questions (2026-11-13)
- DONE - Supabase (keys only in backend/.env): logged-in patients' chats, model results, reports, uploads (files), cases and doctor plans are saved privately per user in the Storage bucket `curanex-uploads` (`<user_id>/<kind>/<time>-<id>.json`). The API keys cannot create tables, so records are JSON objects; `backend/supabase/schema.sql` is the table version for later. `/me/data` + "My saved data" on the Record screen. Tested live: chats 4, uploads 1 (file link opens), reports 1, results 1; no login -> 401.
- DONE - AI Doctor asks 3-5 doctor-style questions with tap-to-answer options on a first vague symptom, then gives the plan. Wider safe OTC list (19), up to 5 options.
- DONE - start_curanex.ps1 (Docker + workers + API + web; prints the iPhone address). Web app talks to the PC that served it.
- DONE (2026-11-13) - AI Nutritionist made real: "Snap a meal" sends the photo to `/v1/food-photo` (free Groq vision model) -> each dish, portion and 15 nutrients (calories, protein, carbs, fat, fibre, sugar, iron, calcium, vitamins C/A/B12/D, folate, sodium, potassium), totals added in code, labelled "AI estimate"; non-food refused. Daily targets from the profile (Mifflin-St Jeor + ICMR-NIN 2020; lower carbs for diabetes). `/v1/meal-plan` makes a 5-meal Indian plan for the diet type, allergies and conditions, with reasons and tips; Regenerate works. "Vitamins and minerals today" card. Tested in the app (coffee photo -> tea with milk 30 kcal; added to diary updated calcium and calories). 148 tests pass.

## Phase 13: smartwatch data from the watch's own app (2026-11-13)
- DECIDED (owner) - No Curanex band. Read what the user's watch app already collected. Android watches only (no Apple Watch). Owner tests with a Nothing watch + Nothing X on an iPhone 17 Pro, free only.
- DONE - iPhone (free): watch app (Nothing X / NoiseFit / Zepp ...) -> Apple Health -> Health app "Export All Health Data" -> upload export.zip at /devices -> `/v1/wearables/apple-health` (`backend/api/wearables.py`, streamed, last 30 days; phone + watch steps never added together). Tested with a 14-day Nothing X sample in the app.
- DONE - Android (needs the APK, not Expo Go): `src/lib/watch.android.ts` reads Health Connect (steps, HR, resting HR, sleep, SpO2, HRV, active kcal, weight, BP) -> `/v1/wearables/sync`. `react-native-health-connect` + `expo-build-properties` (minSdk 26) + Health Connect read permissions in app.json; `eas.json` preview profile builds an APK (`npx eas login` by the owner, then `npx eas build -p android --profile preview`). NOT yet built or tried on a phone.
- DONE - Fixed alert rules (no AI): SpO2 < 92 %, resting HR > 100 / < 40, +10 bpm week on week, < 6 h sleep. Home tiles (steps, resting HR, sleep, SpO2), recommendations and insight use real watch numbers; AI Doctor gets `smartwatch` in its context (uses it, low oxygen + breathless -> see a doctor today); logged-in users' days saved to Supabase ("Watch data" tab).
- DONE - AI Doctor reply language now follows the patient's own words (English / Hinglish / Hindi), not earlier replies. 152 tests pass.
