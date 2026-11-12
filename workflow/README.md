# Curanex workflow folder - start here

This folder lets any Claude Code session (or person) pick up the work exactly where it stopped.
The owner may switch Claude accounts when credits run out, so **keep these files current after every step**.

## Read in this order
1. `STATUS.md` - every phase and task, marked **DONE**, **PENDING** or **WAITING ON USER**. The first PENDING item is the next job.
2. `NEXT_STEPS.md` - the ordered queue with exact commands.
3. `ENVIRONMENT.md` - how to start the servers, run tests, use Kaggle, regenerate the registry.
4. `MODELS.md` - the 50 live models, Verified or not, and the retired ones with reasons.
5. `DOCTOR_PLATFORM_PLAN.md` - the agreed design for the next phase (doctor portal, 80-book RAG).
6. Background: `../CLAUDE.md`, `../docs/MASTER_PROMPT.md`, `../docs/INTEGRATION_REPORT.md` (every model test, pass or fail).

## Rules that must not be broken
- A model is **Verified** only after passing a real outside test (another hospital, country, survey or cycle), with the rules fixed before scoring and the test scored once. The owner has asked to mark failing models Verified; this was refused every time. Never fake metrics.
- Medical safety: red flags are fixed rules (no AI); never say "you have X"; no medicine doses in the patient app; medicines for patients only from the OTC list in `backend/api/orchestrator/care.py`.
- Never commit secrets. The Kaggle token is given by the owner in chat; use it only as an environment variable (`KAGGLE_API_TOKEN`), never save it to a file. Rotate keys (Groq, Vercel, Nia, Kaggle) after the demo.
- Push to GitHub only after the owner says "Yes, push". Commit after each step.
- Do not delete user data. Credit dataset and repo authors.
- Heavy training runs on Kaggle (free GPU/CPU), not the owner's laptop (8 GB RAM, lags).

## How to update this folder
- Change a task's mark in `STATUS.md` the moment it is done (add the date and commit hash).
- Add new tasks at the end of the right phase. Keep `NEXT_STEPS.md` in the order you would do them.
- Regenerate `MODELS.md` after any registry change (command in `ENVIRONMENT.md`).
