# Curanex: instructions for Claude Code

The full phase plan is in `docs/MASTER_PROMPT.md`. Read `README.md`, `docs/BACKEND_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/REPO_AUDIT.md` and `docs/PLAN.md` before changing anything. For backend work, BACKEND_PLAN.md is the source of truth: follow its phases B0–B8 in order and use its Claude Code prompts (§10). The complete step-by-step workflow (build time A0–A10, run time B1–B22, model search) is `docs/WORKFLOW.md`; the prompts are in `docs/CLAUDE_CODE_PROMPTS.md`. To train or download models for backlog repos, follow `docs/TRAINING_PLAN.md`. For hospital-only repos (scan upload, wearables, phone sensors, local alerts, clinic portal), follow `docs/ALL_MODELS_PLAN.md`.

## What exists
- Expo SDK 58 app (React Native + react-native-web): one codebase for Android, iOS and web. Expo Router routes live in `src/app/`.
- Design system in `src/theme.ts` + `src/components/ui.tsx`, taken from the website (`docs/website-index.html`). Reuse these components. Don't introduce new colours, fonts or UI libraries.
- Shared state in `src/store/health.tsx`. Every module reads and writes here. Every AI request sends `getUserContext()`.
- Backend: `backend/main.py` (FastAPI). `/chat` returns the "blocks" contract from ARCHITECTURE.md §6. The app renders it in `src/components/blocks.tsx`.

## Rules
- Must keep working in **Expo Go** and on **web**. Before adding a native library, check it is in Expo Go. Install with `npx expo install`, never with plain npm.
- Check `Platform.OS === 'web'` for things browsers can't do (SMS, camera capture, haptics).
- TypeScript strict. Run `npx tsc --noEmit` and `npm run export:web` before saying a task is done.
- Never put API keys in the app. All LLM, STT and ML calls go through the backend.
- Never load a downloaded .pkl/.joblib/.sav in the API process. Sandbox it, convert it to ONNX, and pin its SHA-256 (BACKEND_PLAN §6).
- SOS and red-flag logic stay deterministic. Never route them through an LLM.
- Medical wording: never "you have X". Every model result shows "Not a diagnosis", what it was trained on and its limitations.
- Trained models: test set used once; never report training/validation scores as results; credit the original repo in the model card.
- Never fabricate model metrics or user stats. Offline/demo output must be labelled "Demo".
- Every screen needs loading, empty and error states, and must work at phone width (390 px) and desktop width (1366 px).

## Next tasks (do them one at a time, then stop and report)
1. Backend: implement the ML router from ARCHITECTURE.md §2–§5 inside `orchestrate()` (registry embeddings, tool retrieval, orchestrator tool loop). Start with the `mlmed/torchxrayvision` adapter.
2. Integrate further models one at a time, using the 45-minute rule in ARCHITECTURE.md §9.
3. Voice: record → `/transcribe` → `/chat` → spoken reply. Text and voice must share one conversation store (`src/store/doctor.tsx`, not built yet; `src/app/voice.tsx` only speaks replies so far).
4. Backend `/food-photo`, `/meal-plan`, `/workout-plan`, and wire them into the Nutrition and Trainer screens (replace the static data in `src/data/content.ts`).
5. Medication reminders with `expo-notifications` (not supported on web: hide them there).
