# Next steps, in order (last updated 2026-11-12)

Mark each one in `STATUS.md` when done.

1. **Replace the 3 Unverified live models** (heart disease, underactive thyroid, chest X-ray layer) with new models that pass an outside test.
   - Ideas not yet tried: other NHANES measured targets (train one cycle, test another), two-hospital clinical datasets like the CKD India -> Bangladesh pair
     (`backend/training/train_ckd_india.py`). Failed ideas are listed in `docs/INTEGRATION_REPORT.md` - do not repeat them.
   - Retire a model by adding it to `RETIRED` in `backend/tools/apply_live_rule.py`, then regenerate the registry.
2. **Outside-test the 3 same-source models**: teeth photo (DentalAI), arm and hand X-ray (MURA), skin lesion (PAD-UFES-20).
   Use `kaggle/make_older_outside_notebook.py` as the pattern (it runs the app's own adapters on Kaggle).
3. **Voice-filled forms** in the AI doctor (patient speaks the values; AI fills `form_values`; read back for a yes).
4. **Phase 9 final verification** (`docs/VERIFICATION.md`): live run of all 50, red-team set, screenshots 390/1366, latency, memory.
5. **Doctor platform** (`DOCTOR_PLATFORM_PLAN.md`) - start when the owner sends the ~80 books:
   accounts + manual approval -> health record storage -> report reading -> books ingestion -> doctor RAG -> portal screens -> share with a doctor -> evaluation.
6. Push to GitHub only when the owner says "Yes, push".
