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
- PENDING - Replace or outside-test the 3 Unverified: heart disease, underactive thyroid, chest X-ray layer.
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
- PENDING - Accounts and roles: patient / doctor (pending -> approved by admin by hand) / admin. Server-side role check on every doctor endpoint.
- PENDING - Health record storage in Postgres (`curanex-postgres`, pgvector): cases, chats, model results, reports, tests ordered, doctor plans; encrypted; audit log; patient consent to share.
- PENDING - Report upload reading (photo/PDF -> values, patient confirms).
- PENDING - Books ingestion: text + OCR, section chunks with book/chapter/page, embeddings, pgvector, hybrid search, re-ranker.
- PENDING - Doctor RAG: retrieve passages -> LLM answer with page references (no match -> LLM answers from its own knowledge); tests to order, likely diseases, medicines with doses, surgery referral; doctor confirms each item.
- PENDING - Doctor portal screens; patient sees only the doctor's confirmed plan.
- PENDING - Evaluation: 100 test questions, retrieval recall and citation faithfulness.
