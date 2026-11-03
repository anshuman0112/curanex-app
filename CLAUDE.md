# Curanex: instructions for Claude Code

Read `README.md`, `docs/ARCHITECTURE.md` and `docs/PLAN.md` before changing anything.

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
- SOS and red-flag logic stay deterministic. Never route them through an LLM.
- Medical wording: never "you have X". Every model result shows "Not a diagnosis", what it was trained on and its limitations.
- Never fabricate model metrics or user stats. Offline/demo output must be labelled "Demo".
- Every screen needs loading, empty and error states, and must work at phone width (390 px) and desktop width (1366 px).

## Next tasks (do them one at a time, then stop and report)
1. Backend: implement the ML router from ARCHITECTURE.md §2–§5 inside `orchestrate()` (registry embeddings, tool retrieval, orchestrator tool loop). Start with the `mlmed/torchxrayvision` adapter.
2. Integrate further models one at a time, using the 45-minute rule in ARCHITECTURE.md §9.
3. Voice: record with `expo-audio` → `POST /transcribe` → `/chat`, in `src/app/voice.tsx`.
4. Backend `/food-photo`, `/meal-plan`, `/workout-plan`, and wire them into the Nutrition and Trainer screens (replace the static data in `src/data/content.ts`).
5. Medication reminders with `expo-notifications` (not supported on web: hide them there).
