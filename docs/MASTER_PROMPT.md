# Master prompt for Claude Code

Paste the block below into Claude Code, started inside the `curanex-app` folder. After each phase it stops and reports. Reply **"continue"** to start the next phase, or tell it what to change.

```
You are building Curanex, an AI healthcare app: an Expo app (iOS, Android, web) in src/ and a Python
FastAPI backend in backend/. The AI Doctor answers by text or voice and must use the 126 open-source
medical model repos listed in docs/repos.txt. I have permission from every repo author, so a missing
licence never blocks a repo.

STEP 0: READ FIRST, CHANGE NOTHING
Read, in this order: CLAUDE.md, README.md, docs/WORKFLOW.md, docs/BACKEND_PLAN.md, docs/ARCHITECTURE.md,
docs/ALL_MODELS_PLAN.md, docs/TRAINING_PLAN.md, docs/REPO_AUDIT.md, docs/model_registry.json,
docs/CLONE_ALL_PROMPT.md, docs/CLAUDE_CODE_PROMPTS.md. Then give me a 15-line summary of what exists,
the AI Doctor workflow (text and voice), how model search works, and the phase plan below. Then stop.

THE GOAL
Every question goes: app → safety check → user context → model search over the registry → the
orchestrator LLM asks for missing inputs or a photo, runs the matching repo model in its worker, and
explains the result → the app shows it, and speaks it if the user talked. If no model fits, the LLM
answers directly. Details: docs/WORKFLOW.md Part B and C.

PHASES (do them in order; stop and report after each; wait for me to say "continue")
1. Backend foundation: BACKEND_PLAN phase B0 (CLAUDE_CODE_PROMPTS.md Prompt 1).
2. Model platform: workers, registry, model search, orchestrator tool loop with a dummy model
   (CLAUDE_CODE_PROMPTS.md Prompt 2).
3. Clone all 126 repos and write docs/INTEGRATION_REPORT.md (docs/CLONE_ALL_PROMPT.md).
4. First real model: mlmed/torchxrayvision end to end in the app (CLAUDE_CODE_PROMPTS.md Prompt 4).
5. Integrate the other wave-1 and wave-2 repos one at a time, READY ones first
   (docs/WORKFLOW.md steps A3–A10 per repo). Report after every 3 models.
6. Retrieval tuning + Model Library shows live status (CLAUDE_CODE_PROMPTS.md Prompt 6).
7. Backlog: sort into tracks (TRAINING_PLAN.md T0), download linked weights (T1), build the training
   recipe (T2), train tabular models (T3). Prepare Kaggle/Colab notebooks for image models (T4), and wait
   for me to run them on the GPU.
8. Hospital-only routes from docs/ALL_MODELS_PLAN.md in this order: C voice screening, D local health
   alerts, A scan upload (2D first), E clinic portal, B wearables.
9. Full end-to-end verification (CLAUDE_CODE_PROMPTS.md Prompt 7).

RULES (never break these)
- Don't break the app: keep /health, /chat, /upload, /transcribe, /models and the /chat "blocks"
  contract working. After every change run backend tests, `npx tsc --noEmit` and `npm run export:web`.
- The app must keep working in Expo Go and on web until phase 8B (wearables needs a development build).
- Repos are untrusted code: clone only into backend/vendor/ (gitignored). Never run their code or install
  their requirements on my machine. Load model files only inside Docker with `--network none`, check the
  SHA-256, convert to ONNX where possible. Never load a pickle in the API process.
- One adapter per model, in the right worker (tabular / torch / tf / later medimg).
- 45-minute rule: if a repo won't produce a prediction in about 45 minutes, mark it backlog with the reason,
  tell me, and move on.
- Metrics are measured by you on held-out data, never copied from a README. No metrics → verified=false.
  A model is switched on only with sensitivity ≥ 0.80 and AUC ≥ 0.80, unless I approve otherwise.
- Medical safety: red-flag messages escalate without calling any AI. Never "you have X". Every result shows
  "not a diagnosis", what it was trained on and its limitations. No medicine doses. SOS stays deterministic.
- Privacy: no health data in logs. Strip identifying DICOM tags. Never store precise location.
- Credit the original repo author in every model card. Retrained models say "trained by the Curanex team
  using the method from <repo>, on <dataset>".
- Commit after each completed step with a clear message.
- If something needs my login, an API key, a paid service, a GPU, or a product decision, stop and ask.
  Don't guess.

REPORT FORMAT (end of every phase)
What you did · test results · models now integrated (with metrics) · what failed and why · what you need
from me · the exact next step.

Start with STEP 0 now.
```
