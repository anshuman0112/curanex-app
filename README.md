# Curanex: mobile + web app

One React Native (Expo) codebase that runs on **Android, iOS and the web**, with a **Python (FastAPI)** backend.
The design follows the Curanex website (`docs/website-index.html`): cream, dark brown and copper colours, with Playfair Display + Onest fonts.

## Run it

### 1. App (phone and web)
```bash
npm install
npx expo start
```
- **Android / iPhone:** install **Expo Go** from the Play Store or App Store and scan the QR code. Your phone and laptop must be on the same Wi-Fi.
- **Web:** press `w` in the terminal, or run `npm run web`.
- **Web build for hosting:** `npm run export:web` creates the static site in `dist/`. Deploy it to any static host (Vercel, Netlify, Firebase Hosting).

### 2. Backend (Python)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export LLM_API_KEY=your_key_here                        # Windows: set LLM_API_KEY=...
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pytest -q                                               # safety tests
```

### 3. Connect the app to the backend
Copy `.env.example` to `.env` and set your laptop's Wi-Fi IP address (find it with `ipconfig` or `ifconfig`):
```
EXPO_PUBLIC_API_URL=http://192.168.1.20:8000
```
Restart `npx expo start`. The Doctor screen header turns from **"Offline demo"** to **"Connected"**.

## What's real and what's demo

| Works now | Demo / placeholder (next steps) |
|---|---|
| Onboarding, health profile and Medical ID, stored on the device | AI answers need `LLM_API_KEY` on the backend |
| Shared health store (`getUserContext`) sent with every AI request | ML model router + adapters (see `docs/ARCHITECTURE.md`) |
| Red-flag → emergency card (runs on device **and** on the backend) | Snap-to-log food recognition (shows a sample result) |
| SOS: hold 3 s, 5 s cancel, location, SMS to contacts, dial 112, hospitals map | Voice: speech output works; microphone → `/transcribe` not wired yet |
| Offline first-aid guides | "Regenerate" meal plan, AI workout plans (static plans for now) |
| Doctor chat with form / upload / model-result cards (the `/chat` blocks contract) | AI form check (camera pose) |
| Guided workout with timer and voice cues; workouts and meals feed the Home score | |
| Model Library of all 126 catalogued models with honest status badges | |

When the backend isn't reachable, the Doctor screen runs an **offline demo**. It shows the flows using sample output that is clearly labelled "Demo data".

## Structure
```
src/app/                Expo Router screens (file = route)
  (tabs)/               Home, Doctor, SOS, Nutrition, Trainer + custom dock
  onboarding.tsx  profile.tsx  models.tsx  voice.tsx  workout.tsx
src/components/         ui.tsx (design system), blocks.tsx (chat cards), icons, ProfileForm
src/store/health.tsx    shared health store + getUserContext()
src/lib/api.ts          backend client + offline demo
src/lib/safety.ts       on-device red-flag rules
src/data/               first aid, plans, model list
backend/main.py         FastAPI: /health /chat /upload /models
docs/                   ARCHITECTURE.md, PLAN.md, model_registry.json, original website
```
