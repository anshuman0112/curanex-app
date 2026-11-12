# Plan: Curanex doctor module (free medical library, no hallucination)

## Context
- The owner wants a doctor-only module: doctors see patient cases, run the Verified prediction models, get help to (1) choose tests,
  (2) identify the likely disease from symptoms + reports, (3) choose medicines (with doses) and decide if surgery/referral is needed.
- The ~80 books could not be found. Decision (owner, 2026-11-12): use free, openly licensed medical sources instead ("Curanex library").
- Hard requirement from the owner: **no hallucination**, but the doctor must still get a list of likely diseases (never a blank answer).
- Patients never see the doctor module; they keep the AI doctor (50 models + OTC list + green/amber/red flag).
- Prerequisite: 2 backend tests still fail from today's model changes (Part 0).

## Part 0 - finish in-progress fixes (small, first)
- `backend/api/reports.py` `REPORT_CHECKS`: remove retired `moizah1_diabetes_prediction` and `yahyazuher_...`; add `_VALUES` patterns
  `triglycerides`, `hdl`, `bun`, `uric_acid`; map {triglycerides, hdl} -> `nhanes_diabetes_any_labs`, {bun, uric_acid} -> `nhanes_low_egfr_labs`.
- `backend/tests/test_report.py`: DIABETES -> `nhanes_diabetes_any_labs`, add TG/HDL lines to `LAB`.
- `backend/tests/test_orchestrator.py:176`: torchxrayvision is now Unverified -> assert `verified is False`.
- Full pytest green, commit, mark DONE in `workflow/STATUS.md`.

## 1. Who can do what
| Role | How they get it | Can do |
|---|---|---|
| Patient | Signs up in the app | AI doctor, own record, upload reports, share a case with a doctor, see the doctor's confirmed plan |
| Doctor (pending) | Signs up as doctor | Sees "waiting for approval" only |
| Doctor (approved) | Admin approves by hand | Shared cases, models, library, tests, diagnosis, prescriptions, confirm/reject |
| Admin (owner) | Set in config | Approve/reject/suspend doctors, see audit log |
Every doctor/admin endpoint checks the role on the server (FastAPI dependency `require_role("doctor")`). Patient tokens get 403.

## 2. Free medical library (replaces the books)
| Source | Used for | Licence (stored on every chunk, shown in citations) |
|---|---|---|
| StatPearls (NCBI Bookshelf, ~9,000 chapters incl. symptom chapters) | Diseases, differentials, tests, treatment, when to refer/operate | CC BY-NC-ND 4.0 |
| ICMR Standard Treatment Workflows (India) | Indian first-line tests, treatment, referral | Free Govt. of India PDFs |
| WHO guidelines + WHO Model Formulary / Essential Medicines List | Protocols, essential drug use | CC BY-NC-SA 3.0 IGO (most) |
| DailyMed / openFDA drug labels | **Doses, contraindications, interactions** (the only source of doses) | Public domain |
| MedlinePlus | Plain-language text for the patient view | Public domain |
Exact download URLs and licences are verified when each fetcher is written; anything whose licence forbids this use is left out.

## 3. Code structure (new)
```
backend/knowledge/            # offline: build the library
  fetch_statpearls.py, fetch_icmr.py, fetch_who.py, fetch_druglabels.py, fetch_medlineplus.py
  ingest.py                   # parse HTML/XML/PDF -> section chunks + metadata -> embeddings -> Postgres
backend/api/knowledge/        # online: search + grounded answers
  search.py                   # hybrid search: pgvector + Postgres full-text, merged, top-k, relevance score
  answer.py                   # tests / differential / plan builders (structured JSON from the LLM)
  verify.py                   # evidence check: every claim must be supported by its cited chunk
  drugs.py                    # medicine + dose lookup from drug labels only (copied, never generated)
backend/api/auth/             # users, roles, sessions, doctor approval
backend/api/records/          # health record: cases, messages, model results, reports, orders, plans, audit
backend/api/routers/doctor.py # /doctor/* endpoints (role-guarded)
backend/api/routers/admin.py  # /admin/doctors (approve / reject / suspend)
src/app/doctor/               # doctor portal screens (Expo Router, same design system)
src/app/record.tsx            # patient's own record + "Your doctor has advised"
backend/tests/knowledge_cases.json + test_knowledge_*.py, test_auth_roles.py, test_records.py
```
Reuse: embedder `api/orchestrator/retriever.get_embedder`; LLM `api/clients/llm.chat_completion` (Groq gpt-oss-120b, temperature 0, JSON mode);
report reading `api/reports.py` (`find_values`, OCR); model runs `api/orchestrator/tools.py`; red flags `api/safety`.

## 4. Data (Postgres `curanex-postgres`, pgvector)
- `users` (id, role, name, contact, status pending/approved/suspended, created_at)
- `cases` (id, patient_id, created_at, status), `case_shares` (case_id, doctor_id, granted_at, revoked_at)
- `messages`, `model_results` (model_id, inputs hash, label, scores, flag), `reports` (file ref, extracted values, patient-confirmed)
- `orders` (tests ordered by doctor), `plans` (diagnosis, medicines, referral, confirmed_by, confirmed_at)
- `knowledge_chunks` (id, source, title, section, url/page, licence, text, embedding vector(384), tsvector)
- `audit_log` (who, what, case_id, when). Reports and records encrypted at rest; only de-identified text goes to the LLM.

## 5. Doctor flows
**Stage 1 - which tests?** Doctor opens a shared case -> symptoms, history, model results -> library search on the symptoms ->
test suggestions, each with the reason and its citation -> doctor ticks/edits -> order appears in the patient's app.
**Stage 2 - which disease, what treatment?** Patient uploads report -> values read and confirmed by the patient -> Verified models re-run ->
differential (section 6) -> for a chosen disease: treatment plan from the library + doses from drug labels + "needs surgical/specialist opinion"
when the library says so -> doctor confirms/edits/rejects each line -> patient sees only "Your doctor has advised…".

## 6. No-hallucination design (the core)
1. **AI proposes, library verifies.** The LLM lists up to 8 candidate diseases from the case. For each candidate the system searches the library
   and keeps the passages that describe that disease with the patient's findings.
2. **Supported diseases** (passage found, findings match): ranked list with "why" quotes from the passage + citation [source, section].
3. **Not supported by the library**: shown in a separate short list "Also consider (not confirmed by the library)" - name only,
   no treatment, no doses. So the doctor always gets likely diseases, but nothing unverified is presented as fact.
4. **Every sentence is checked** (`verify.py`): each claim must carry a citation, and the cited chunk must contain the supporting text
   (key terms and numbers present; numbers must match exactly). Failing sentences are removed before display.
5. **Doses are never written by the AI.** `drugs.py` copies the dose, contraindications and interactions text from the official drug label
   and shows it as a quote with the label link. If no label is found, no dose is shown.
6. **Model results are never changed** by the LLM; they are shown as computed, with the Verified status and test numbers.
7. Temperature 0, structured JSON output, emergency red-flag rules unchanged (no AI), doctor confirmation required for every item, full audit log.
8. Uncertainty is visible: each supported disease shows how many passages support it and which findings were matched.

## 7. Build phases (each ends with tests + commit + workflow/STATUS.md update)
1. Part 0 fixes.
2. Auth + roles + admin approval screen; tests prove patients and pending doctors get 403 on /doctor/*.
3. Health record tables, case sharing with consent, audit log; patient record screen.
4. Library: fetchers + ingest for ICMR + drug labels + ~500 StatPearls chapters for common conditions first; then the rest.
5. Search (`search.py`) + eval set (60+ questions with expected source sections); tune the relevance cut-off.
6. Grounded answers: tests (stage 1), differential with verification (section 6), treatment + doses (drugs.py).
7. Doctor portal screens (case list, case view, tests, differential, plan, confirm) at 390 px and 1366 px; patient "Your doctor has advised" view.
8. End-to-end run on 5 cases (diabetes, TB, anaemia, kidney disease, fever) and hallucination audit.

## Verification
- `pytest` all green; `npx tsc --noEmit`; `npm run export:web`.
- Retrieval: recall@5 >= 0.85 on `knowledge_cases.json`.
- Faithfulness audit: 30 generated answers, every displayed sentence traced to its cited chunk (target 100%; any miss is a bug).
- Dose audit: every dose shown matches the drug label text exactly.
- Role tests: patient token and pending doctor token -> 403 on every /doctor and /admin route.
- Live demo on the 5 cases with screenshots; results written to `docs/VERIFICATION.md` and `workflow/STATUS.md`.
