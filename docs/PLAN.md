# Curanex — Mobile App Build Plan

> Hand this file to Claude Code. Build **one phase at a time**. Do not start a phase until the previous phase's "Done when" checks pass on a real device or emulator.

---

## 0. Product framing (read first)

Curanex is a **health guidance and triage** app, not a diagnostic device. Every AI output that touches symptoms, images or medication must:
- be phrased as "possible causes / what to do next", never "you have X"
- show a confidence level and a "see a doctor if…" line
- escalate red-flag symptoms (chest pain, stroke signs, breathing difficulty, heavy bleeding, suicidal thoughts) straight to the SOS flow instead of continuing a chat

**The real differentiator is the shared health profile.** Four modules sitting in one app is not unique (HealthifyMe, Practo, Ada, Cult.fit each cover parts). What is unique is that they *talk to each other*:
- Doctor knows what you ate and how you trained this week
- Nutrition plan adapts to a condition the Doctor module flagged (e.g. high BP → low sodium)
- Trainer avoids exercises that conflict with a reported injury or condition
- SOS sends your medical ID (allergies, meds, blood group, conditions) to responders

Every phase below must read from and write to the same `HealthProfile` + `HealthEvents` store.

---

## 1. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Mobile | **Expo (React Native) + TypeScript + Expo Router** | One codebase for Android/iOS; Claude Code is very productive with it |
| UI | NativeWind (Tailwind) + custom component library | Fast, consistent theming, dark mode |
| State / data | Zustand (local state) + TanStack Query (server state) | Simple, cache-friendly |
| Backend | **FastAPI (Python)** | ML models live in Python anyway |
| Auth / DB / Storage | **Supabase** (Postgres + Auth + Storage + Row Level Security) | Free tier, RLS for health data |
| LLM | Claude API (or any LLM API) via backend only — never call from the app with a key in the bundle | Security |
| Speech | `expo-speech` (TTS) + speech-to-text via Whisper API or on-device recognizer | Voice doctor |
| Vision (food, reports) | Multimodal LLM through backend | No custom training needed |
| CNN (medical image demo) | PyTorch/TensorFlow model trained on a public dataset, served by FastAPI | Shows the DL/CNN work for the project |
| Pose (trainer) | MoveNet / MediaPipe Pose (on-device) | Rep counting, form checks |
| Maps | `react-native-maps` + Google Places API | Nearby hospitals |
| Location / SMS | `expo-location`, `expo-sms`, `expo-linking` (tel:) | SOS |
| Notifications | `expo-notifications` | Meds, water, workouts |
| Wearables (later) | Health Connect (Android) / HealthKit (iOS) — needs a dev build | Steps, heart rate, sleep |

> Note: camera pose detection, Health Connect and HealthKit require an **Expo development build** (not Expo Go). Set that up at Phase 0.

---

## 2. Information architecture (screens)

### Bottom tabs
`Home` · `Doctor` · `SOS` (center, red, raised) · `Nutrition` · `Trainer`
Profile & settings live behind the avatar on Home.

### Onboarding
1. Welcome / value carousel (3 slides)
2. Sign up (phone OTP or email)
3. Consent screen — what data is collected, why, toggle per category (health records, location, camera, mic)
4. Health profile wizard: age, sex, height, weight, blood group, conditions, allergies, current medications, diet type (veg / non-veg / vegan / Jain / eggetarian), activity level, goals
5. Emergency contacts (min 1) + permission requests explained in plain language
6. Medical ID preview

### Home (dashboard)
- Greeting + daily health score (composite of sleep, steps, nutrition adherence, meds taken)
- Today cards: next medication, meals logged vs plan, today's workout, water
- AI insight card ("Your sodium has been high 3 days in a row — your BP readings went up too")
- Quick actions: Talk to Doctor, Log meal, Start workout, Upload report
- Vitals strip (manual entry or wearable): HR, BP, SpO₂, glucose, weight

### Doctor
- **Chat** — text conversation with streaming replies, suggested follow-up chips
- **Voice mode** — full-screen orb/waveform, push-to-talk + hands-free, live transcript, interrupt support
- **Symptom checker** — structured flow (body map tap → symptom → duration → severity) → triage result: *self-care / see GP / urgent care / emergency*
- **Report reader** — upload a lab report PDF/photo → plain-language explanation, out-of-range values highlighted, trend vs previous reports
- **Image analysis** — skin, X-ray (research demo), with heavy disclaimer
- **Health records** — timeline of reports, prescriptions, visits, vitals
- **Medication manager** — schedule, reminders, adherence streak, drug-interaction warning
- **Risk assessment** — diabetes / heart disease risk score from profile + vitals (explainable: show which factors drove it)
- **Consult summary** — generate a one-page summary to show a real doctor

### SOS
- One giant button: **press and hold 3 s** (prevents pocket triggers) → 5 s cancel countdown
- On trigger: call 112, SMS live location link + medical ID to all emergency contacts, start location sharing
- Quick-dial tiles: 112 (national emergency), 108 (ambulance), 102, women's helpline, custom
- Nearby hospitals map with distance, open/24h, call + navigate buttons
- **Offline first-aid guides** (CPR, choking, burns, bleeding, seizure, snakebite, heart attack, stroke FAST) — bundled in the app, work with no internet
- Medical ID on lock screen (Android widget / iOS Medical ID instructions)
- Later: fall detection (accelerometer), shake-to-SOS, voice trigger

### Nutrition
- Today's plan (meals with macros), swap a meal
- **Snap-to-log** — photo of plate → identified dishes + portion estimate + calories/macros, editable
- Search-to-log with Indian food database
- Weekly plan generator (respects diet type, conditions, budget, cuisine region)
- Water tracker
- Grocery list auto-generated from the week's plan
- Condition-aware rules (diabetes → low GI, CKD → potassium caution, PCOS, hypertension)
- Weekly report: adherence, nutrient gaps

### Trainer
- Plan for the week (home / gym / yoga), adapts to goals and conditions
- Exercise library with animations and target muscles
- **Live workout mode** — camera pose detection, rep counter, form feedback ("knees going past toes"), rest timer, voice cues
- Progress: workout streak, volume, weight trend, before/after photos (stored privately)
- Recovery check-in (soreness, sleep) → adjusts tomorrow's intensity

### Profile & settings
- Health profile edit, emergency contacts, medical ID
- Connected devices (Health Connect / HealthKit)
- Language (English, Hindi, Kannada, Tamil, Telugu…) — voice too
- Data & privacy: export my data, delete my account, consent toggles
- Theme, units, notifications

---

## 3. Data model (Supabase / Postgres)

```
users(id, phone, email, created_at)
health_profiles(user_id, dob, sex, height_cm, weight_kg, blood_group, diet_type, activity_level, goals[], updated_at)
conditions(id, user_id, name, icd10_code?, since, severity, source[self|doctor|ai_flagged])
allergies(id, user_id, substance, reaction)
medications(id, user_id, name, dose, schedule_json, start, end)
med_logs(id, medication_id, taken_at, status[taken|skipped])
emergency_contacts(id, user_id, name, phone, relation, priority)
vitals(id, user_id, type, value, unit, measured_at, source)
health_events(id, user_id, module[doctor|nutrition|trainer|sos], type, payload_json, created_at)   -- shared timeline
chat_sessions(id, user_id, mode[text|voice], started_at, triage_level)
chat_messages(id, session_id, role, content, created_at)
reports(id, user_id, file_path, kind, extracted_json, ai_summary, created_at)
meals(id, user_id, eaten_at, items_json, kcal, protein, carbs, fat, photo_path?)
meal_plans(id, user_id, week_start, plan_json)
workouts(id, user_id, started_at, plan_ref, exercises_json, duration_s)
sos_events(id, user_id, triggered_at, lat, lng, contacts_notified[], cancelled)
consents(user_id, category, granted, updated_at)
```
**Row Level Security on every table: `user_id = auth.uid()`.** Storage buckets private.

The `health_events` table + a `buildUserContext(user_id)` backend function is what makes modules aware of each other — every LLM prompt in every module gets this context summary.

---

## 4. Backend API (FastAPI)

```
POST /ai/chat            stream; body: session_id, message  → uses buildUserContext + safety layer
POST /ai/voice           audio in → transcript → chat → TTS text out
POST /ai/triage          structured symptoms → {level, reasons, next_steps}
POST /ai/report          file → extracted values + explanation
POST /ai/food-photo      image → dishes, portions, nutrition
POST /ai/meal-plan       → week plan json
POST /ai/workout-plan    → week plan json
POST /ml/xray            image → class probabilities (CNN demo)
POST /ml/risk            profile+vitals → risk scores + feature contributions
GET  /places/hospitals   lat,lng → nearby list
```

### Safety layer (applies to every /ai route)
1. **Red-flag detector** runs before the LLM (keyword + small classifier). If triggered → return `{escalate: "sos", reason}`; app shows emergency card with call buttons.
2. System prompt forbids definitive diagnoses, dosing changes, and prescription-only drug recommendations.
3. Output post-check: must include a "when to see a doctor" line; strip any dosage instructions.
4. Log every AI response with model version for audit.

---

## 5. Phased build plan

Each phase ends with something demoable. Commit after each.

### Phase 0 — Foundation (≈3–4 days)
- Expo app with TypeScript, Expo Router, NativeWind, dev build configured
- Design tokens (colors, type scale, spacing, radius), light + dark theme
- Base components: Button, Card, Input, Chip, Sheet, Toast, EmptyState, Skeleton, Disclaimer
- Supabase project, schema migration from §3, RLS policies
- FastAPI skeleton with `/health` endpoint, env-based config, CORS
- **Done when:** app runs on Android emulator, can sign up with OTP/email, empty tabs render, backend `/health` returns 200

### Phase 1 — Onboarding + profile + Home shell (≈4 days)
- Full onboarding wizard (§2), consent screen, emergency contacts
- Home dashboard with real data from profile, placeholder cards for other modules
- **Done when:** a new user completes onboarding and sees their profile reflected on Home; data persists after app restart

### Phase 2 — SOS (do this early; it's the most life-critical and the simplest to make reliable) (≈4 days)
- Hold-to-trigger button + cancel countdown
- Call 112 via `tel:`, SMS with Google Maps link + medical ID to contacts
- Nearby hospitals map
- Offline first-aid guides bundled as local JSON/markdown
- **Done when:** with airplane mode ON, first-aid guides open; with network, SOS sends SMS with correct location to a test contact; accidental taps do not trigger

### Phase 3 — Doctor: text chat + triage (≈6 days)
- Streaming chat UI, session history
- `buildUserContext` + safety layer
- Symptom checker with body map → triage levels
- Red-flag → SOS escalation card
- **Done when:** typing "crushing chest pain spreading to left arm" shows the emergency card instead of a chat reply; normal questions get context-aware answers referencing the user's conditions

### Phase 4 — Doctor: voice (≈4 days)
- Push-to-talk recording → STT → chat → TTS playback
- Live transcript, interrupt/stop, hands-free toggle
- Hindi + English minimum
- **Done when:** a full voice Q&A round trip works in under ~4 s on a normal connection, and red-flag escalation works by voice too

### Phase 5 — Nutrition (≈6 days)
- Indian food database seeded (see §7)
- Search-to-log, snap-to-log, water tracker
- Weekly plan generator respecting diet type + conditions
- Grocery list
- **Done when:** photo of a plate of dal-rice-sabzi returns editable items with macros; a diabetic profile's plan visibly differs from a non-diabetic one

### Phase 6 — Trainer (≈7 days; riskiest phase)
- Exercise library + weekly plan generator (condition-aware)
- Guided workout mode with timers and voice cues (ship this first)
- Then: camera pose detection + rep counting for 3 exercises (squat, push-up, bicep curl)
- **Fallback if pose is too slow on target devices:** keep guided mode, mark pose as beta
- **Done when:** 10 squats in front of the camera count as 10 (±1)

### Phase 7 — Records, reports, meds, image ML (≈6 days)
- Report upload → extraction → explanation → trends
- Medication manager + reminders + adherence
- CNN image demo endpoint (§6) with research-only labelling
- Risk score with explainability
- **Done when:** uploading two CBC reports shows a trend chart; a missed med triggers a reminder

### Phase 8 — Integration layer (the differentiator) (≈4 days)
- Home AI insight card driven by cross-module `health_events`
- Doctor references diet/workout history; Nutrition/Trainer react to new conditions
- Consult summary PDF export
- **Done when:** adding "hypertension" as a condition changes tomorrow's meal plan, today's workout intensity, and the Doctor's answers — without the user repeating it

### Phase 9 — Polish & hardening (≈5 days)
- Localization, accessibility (font scaling, screen reader labels, contrast), offline states, error states
- Data export + account deletion
- Wearable sync (Health Connect first)
- Crash reporting, basic analytics (no health data in analytics)
- **Done when:** app passes a manual accessibility pass and every screen has loading / empty / error states

---

## 6. ML / DL components (for the "CNN & deep learning" part of the project)

Be honest about what's trained vs what's an API:

| Feature | Approach | Trained by you? |
|---|---|---|
| Chest X-ray classifier (normal / pneumonia) | Fine-tune a pretrained CNN (ResNet/EfficientNet/DenseNet) on a public chest X-ray dataset; report accuracy, sensitivity, specificity, confusion matrix; Grad-CAM heatmap overlay in app | **Yes** — this is your CNN showcase |
| Skin condition (optional second model) | Same pipeline on a public dermatology dataset; note dataset skin-tone bias as a limitation | Yes |
| Diabetes / heart risk | Gradient boosting or logistic regression on public tabular datasets + SHAP explanations | Yes |
| Red-flag classifier | Small text classifier + keyword rules | Yes |
| Chat, food photo, report reading | Multimodal LLM API | No (integration) |
| Pose estimation | Pretrained MoveNet/MediaPipe + your own rep-counting logic (joint-angle state machine) | Logic yes, model no |

Every ML screen shows: model name, what it was trained on, known limitations, "not a diagnosis".

---

## 7. Data sources to check

- Indian food composition: **IFCT 2017** (National Institute of Nutrition) — verify licence before bundling
- Exercise library: build your own JSON (name, muscles, equipment, contraindications, cues)
- First-aid content: base on a reputable source (e.g. Red Cross / St John Ambulance guides), cite in-app
- Public medical image and tabular datasets: verify each dataset's licence permits your use

---

## 8. Privacy, legal & safety checklist

- [ ] India's **Digital Personal Data Protection Act, 2023** — explicit consent, purpose limitation, right to erasure
- [ ] Health data encrypted at rest (Supabase) and in transit (HTTPS only)
- [ ] No API keys in the mobile bundle
- [ ] No health data sent to analytics/crash tools
- [ ] Clear "not a medical device / not a diagnosis" positioning; any real launch would need a regulatory review (software that diagnoses can fall under medical-device rules)
- [ ] Future: ABDM / ABHA integration for importing real health records
- [ ] SOS tested on real devices on both platforms; never rely on the AI for SOS logic

---

## 9. Testing

- Unit tests for triage rules, red-flag detector, rep counter, nutrition calculations
- API tests for every backend route (pytest)
- E2E smoke tests (Maestro) for: onboarding, SOS trigger + cancel, chat red-flag escalation, meal log
- A **red-team prompt list** (50+ risky prompts: overdose questions, self-harm, "should I stop my BP meds", pediatric dosing) run against the chat after every prompt change

---

## 10. Advanced features backlog (after Phase 9)

- Wearable anomaly alerts (resting HR spike, SpO₂ drop) → nudge or SOS
- Fall detection + auto-SOS if no response in 30 s
- Doctor booking / teleconsult handoff with the consult summary attached
- Family mode: caregiver dashboard for elderly parents (meds adherence, SOS alerts)
- Menstrual / pregnancy tracking with condition-aware nutrition
- Mental wellbeing check-ins (mood, sleep, stress) with clear crisis-line routing
- Offline LLM for basic Q&A on low-connectivity areas
- Vernacular voice-first mode for low-literacy users
- Pharmacy price comparison / generic alternatives (informational)
- Gamification: streaks, challenges with friends
