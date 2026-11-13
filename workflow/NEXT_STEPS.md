# Next steps, in order (last updated 2026-11-14)

Mark each one in `STATUS.md` when done.

1. ~~Replace the 3 Unverified live models~~ **DONE 2026-11-14**: replaced by low muscle mass, heart strain (NT-proBNP) and heart muscle injury (troponin), all Verified on later NHANES cycles. 50 live, 50 Verified. Retire a model by adding it to `RETIRED` in `backend/tools/apply_live_rule.py`, then regenerate the registry.
2. **Outside-test the 3 same-source models**: teeth photo (DentalAI), arm and hand X-ray (MURA), skin lesion (PAD-UFES-20).
   Use `kaggle/make_older_outside_notebook.py` as the pattern (it runs the app's own adapters on Kaggle).
3. **Voice-filled forms** in the AI doctor (patient speaks the values; AI fills `form_values`; read back for a yes).
4. **Phase 9 final verification** (`docs/VERIFICATION.md`): live run of all 50, red-team set, screenshots 390/1366, latency, memory.
5. **Doctor platform** (`DOCTOR_PLATFORM_PLAN.md`) - start when the owner sends the ~80 books:
   accounts + manual approval -> health record storage -> report reading -> books ingestion -> doctor RAG -> portal screens -> share with a doctor -> evaluation.
6. Push to GitHub only when the owner says "Yes, push".
