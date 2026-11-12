# Doctor platform plan (agreed with the owner on 2026-11-12)

## Two separate areas
**Patient app (anyone)** - the AI doctor only:
- Voice or text. Fixed red-flag rules first (emergency -> SOS / call 108).
- RAG over the 50 live models: retrieve the matching model(s) -> run the prediction -> the LLM explains the result with its own knowledge.
  The LLM may add context but can never overrule the model's result.
- Traffic light on results: green "No problems found in these results", amber "Please see a doctor", red = emergency.
- Quick-relief medicines only from the fixed OTC list (no doses, "use as on the pack"); never prescription drugs.
- Always says which doctor to see and how soon. "Share with a doctor" sends the case to the doctor channel (patient consent).

**Doctor portal (approved doctors only)**:
- Sign-up -> account "pending" -> admin approves by hand (no registration-number check for now; may be added later).
- Every doctor endpoint checks the role on the server. Books, doses and prescriptions never reach the patient app.
- Sees full model details, the books RAG, test suggestions, medicines with doses, surgery referral. Confirms each item.

## Two-stage clinical flow (books RAG)
1. Stage 1 (before tests): symptoms + history -> books RAG suggests tests (with page references) -> doctor orders tests -> patient's app shows the order.
2. Patient does the tests and uploads the report (photo/PDF) -> values read and confirmed by the patient -> Verified models run.
3. Stage 2 (after tests): report values + model results -> books RAG suggests likely diseases (ranked), treatment plan (medicines with doses as in the books,
   lifestyle, follow-up tests, surgical opinion when relevant) -> doctor confirms/edits/rejects -> patient sees only the doctor's confirmed plan.

## RAG design
- Retrieve: book passages (text + OCR, split by section, tagged book/edition/chapter/page) -> embeddings (local, free) in pgvector + keyword search -> re-ranker -> top 5-8.
- Generate: the passages + doctor question + de-identified patient summary go to the LLM (Groq gpt-oss-120b, free). Book-backed points carry a page reference.
- No good match (relevance below a tuned cut-off): the LLM answers from its own knowledge (no banner; those points simply have no page reference).
- Books stay on the owner's machine only (copyright); free sources (ICMR STGs, WHO guidelines, StatPearls) can be added.

## Storage (Curanex app, Postgres `curanex-postgres`)
- One health record per patient: chats, model results, uploaded reports and values, tests ordered, doctor's diagnosis and plan.
- Patient sees their own record (not the doctor's working notes). Doctor sees only cases shared with them. Admin manages doctor approval.
- Encrypted at rest; every view/change logged (who, when); patient can stop sharing; only de-identified data goes to the LLM.

## Build order and done-checks
1. Roles + manual approval (tests: patient blocked from doctor endpoints; pending doctor sees nothing).
2. Health record tables + consent sharing.
3. Report reading (photo/PDF -> values) with patient confirmation.
4. Books ingestion + search (needs the books).
5. Doctor RAG answers (stage 1 tests, stage 2 diagnosis/plan) with page references.
6. Portal screens (390 px and 1366 px), patient "Your doctor has advised" view.
7. Evaluation: 100 questions with known answers - retrieval recall@k and citation faithfulness; report the numbers.
