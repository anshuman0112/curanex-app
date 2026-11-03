# Claude Code prompts: clone the model repos and connect them to the AI Doctor

Paste these into Claude Code **one at a time, in order**. After each one, Claude Code stops and reports. Check the report, then paste the next.

---

## Before you start (one-time setup on your computer)

1. Unzip `curanex-app.zip` and open a terminal in the `curanex-app` folder.
2. Install: **Node 20+**, **Python 3.11**, **Git**, and **Docker Desktop** (Docker is needed to load untrusted model files safely).
3. Free disk space: about **15–20 GB** (repos, model weights, Python environments).
4. Create `backend/.env` from `backend/.env.example` and fill in at least:
   - `LLM_API_KEY=` (required for the doctor's answers)
   - `OPENAI_API_KEY=` (for voice speech-to-text; optional)
5. Start Claude Code in the `curanex-app` folder: `claude`
6. Claude Code reads `CLAUDE.md` automatically. The prompts below refer to files in `docs/`.

**How the result will work (the workflow):**
```
User types or speaks
 → Safety check (red flags → emergency card)          [already built]
 → User context (profile, meals, workouts)            [already built]
 → Find matching model in the registry (semantic search)        ← Prompts 2–3
 → If a model fits: ask for missing inputs / photo,
   run the repo model in its worker, get prediction + explanation ← Prompts 4–6
 → If no model fits: LLM answers directly             [already built]
 → LLM explains the result safely → app shows a card and speaks it if voice
```

---

## Prompt 0: context (paste this first, every new Claude Code session)

```
You are working on Curanex, an AI healthcare app (Expo app in src/, FastAPI backend in backend/).
Before doing anything, read: CLAUDE.md, README.md, docs/BACKEND_PLAN.md, docs/ARCHITECTURE.md,
docs/REPO_AUDIT.md and docs/model_registry.json. Summarise in 10 lines what exists today, what
the AI Doctor workflow is, and which parts are not built yet. Do not change any files yet.

Ground rules for this whole project:
- The app depends on the backend routes /health, /chat, /upload, /transcribe, /models and on the
  /chat "blocks" contract (ARCHITECTURE.md §6). Never break them. Run backend tests and
  `npx tsc --noEmit` after every change.
- Model repos are UNTRUSTED code. Clone them only into backend/vendor/ (gitignored). Never import
  vendor code from the API. Never load a .pkl/.joblib/.sav/.pt/.h5 in the API process.
  Load them only inside a Docker container with no network, then convert to ONNX where possible.
- Every model gets an adapter in the correct worker (tabular / torch / tf) implementing
  ModelAdapter (BACKEND_PLAN §3.1).
- Never mark a model integrated or verified without a passing smoke test. Never invent metrics.
- 45-minute rule per repo: if it does not produce a prediction in ~45 minutes of work, stop,
  record why in the registry, and move on.
- Medical wording: never "you have X". Results always show "not a diagnosis", trained_on, limitations.
- Work one prompt at a time. At the end of each, stop and report: what you did, test results,
  anything that failed, and what you need from me.
```

---

## Prompt 1: backend foundation (BACKEND_PLAN phase B0)

```
Do BACKEND_PLAN.md phase B0.
Restructure backend/ into the layout in BACKEND_PLAN §8 (api/, workers/, registry/, tools/, tests/, vendor/)
WITHOUT breaking the existing routes: keep `uvicorn main:app` working, or update README and app
docs if the entry point changes. Move safety code to api/safety/ and keep all existing tests passing.
Add: pydantic-settings config reading backend/.env, structured logging (no health data in logs),
docker-compose.yml with services api, worker-tabular, worker-torch, worker-tf, redis
(postgres optional for now), Dockerfiles per worker using backend/requirements/*.txt,
ruff + pytest config.
Workers can be empty "hello" services for now (GET /health, GET /models returning []).
Run: pytest, `docker compose build`, `docker compose up`, curl every /health.
Stop and report.
```

## Prompt 2: model platform (workers + registry + retrieval + orchestrator) (phase B2)

```
Do BACKEND_PLAN.md phase B2, without any real model yet.
1. workers/common/: ModelAdapter base + Prediction dataclass (ARCHITECTURE §4) and a shared
   FastAPI worker server: GET /health, GET /models, POST /predict/{model_id}; lazy loading,
   LRU of max 5 loaded models, 10 s timeout per prediction, input validation errors as 422.
2. A "dummy_echo" adapter in worker-tabular so the pipeline can be tested end to end.
3. api/: registry loader for registry/model_registry.json (copy from docs/); only models with
   integrated=true are routable.
4. api/orchestrator/retriever.py: embed each integrated model's `description` with
   sentence-transformers all-MiniLM-L6-v2 (cache to disk), cosine top-k (RETRIEVAL_TOP_K),
   threshold (RETRIEVAL_THRESHOLD). No vector DB.
5. api/orchestrator/loop.py: LLM tool-use loop (ARCHITECTURE §5): retrieved models become tools
   run_<model_id> with their input_schema, plus request_user_input and request_image.
   Max 3 tool calls. request_user_input → form_request block; request_image → upload_request block;
   model call → worker /predict → model_result block + LLM explanation text block.
   Pre-fill inputs marked from_profile using the request context. Respect `via: voice`
   (short spoken reply). Safety pre-check and post-check stay on every path.
6. Wire it into /chat so behaviour is unchanged when no model is integrated (LLM answers directly).
7. Tests: contract tests for every block type, a test that dummy_echo is routed for a matching
   phrase and NOT for "healthy breakfast", and a red-flag test that never reaches the LLM.
Stop and report with test output.
```

## Prompt 3: clone the repos (wave 1 and wave 2 only)

> To download **all 126** repos (metadata only for backlog and hospital-only), use the Prompt 3 variant in `docs/WORKFLOW.md` Part E instead.

```
Clone the model repos into backend/vendor/ and produce a detailed integration report.
Only clone repos where docs/model_registry.json has wave = "wave-1" or "wave-2" (23 repos).
Do NOT clone "hospital-only" repos. Do NOT clone "backlog" repos unless I ask.

For each repo:
- `git clone --depth 1 <url> backend/vendor/<model_id>`; record the commit SHA.
- Do not run any code from the repo and do not pip install its requirements on my machine.
- Inspect: the weight files (path, size, format), how the model is trained/loaded (read the
  scripts/notebooks), the exact input features with units and order, the preprocessing (scalers,
  encoders, image size/normalisation), the output classes, the licence, and the dataset it was
  trained on.
- Compute SHA-256 of each weight file.
Write docs/INTEGRATION_REPORT.md with one section per repo: weights, features, preprocessing,
labels, licence, dataset, which worker it belongs to (tabular / torch / tf), risks, and a verdict:
READY / NEEDS WORK (say what) / SKIP (say why).
Also update the registry `audit` fields with weight sizes, SHA-256 values and commit SHA.
Stop and report the table of verdicts.
```

## Prompt 4: first real model, torchxrayvision (do this one first)

```
Integrate mlmed/torchxrayvision into worker-torch following BACKEND_PLAN §3.2 steps 3–8.
- Use the pip package torchxrayvision (weights download on first load; cache them in the
  worker image or a volume so it works offline afterwards).
- Adapter: accept an uploaded image_id (the API passes the image bytes to the worker), convert to
  grayscale, resize to the model's expected size, normalise exactly as the library requires,
  run a DenseNet "all" model, return the top findings with probabilities.
- Reject images that are clearly not chest X-rays (simple checks: grayscale-ish, aspect ratio,
  plus a warning in Prediction.warnings). Add Grad-CAM for the top finding as a heatmap image_id.
- Fill the registry entry: input_schema, output_labels, trained_on, limitations, licence,
  description in everyday words ("chest x-ray", "lung scan", "pneumonia", "cough and fever x-ray").
- Smoke test with a sample chest X-ray (download a public sample into backend/tests/fixtures/
  and tell me its source and licence).
- Evaluate on a small public labelled set if feasible (tell me which), record real metrics;
  otherwise leave metrics null and say so.
- Set integrated=true and verified=true only if the tests pass.
- Then run the app against the backend: "here is my chest X-ray" → upload card → result card +
  explanation. "What's a healthy breakfast?" → no model.
Stop and report.
```

## Prompt 5: onboard one model at a time (repeat for each repo)

Paste once per repo, replacing `<REPO>`. Suggested order:
1. `Moizah1/Diabetes-Prediction` (tabular, includes a SHAP explainer)
2. `Sneha798/Heart-Disease-Prediction-Project` (tabular)
3. `yahyazuher/AI-Liver-Disease-Diagnosis-System` (tabular, several models)
4. `singh2010nidhi/Heart-Failure-Prediction-using-MSAzure` (already ONNX)
5. `joystonmenezes/xray-vision` (bone X-ray, already ONNX)
6. `OmerKurtulus/skin-disease-classification` (tf worker, phone photo)
7. `AndreyGermanov/yolov8_caries_detector` (torch worker, GPL: server-side only)
8. then wave 2, in the order of docs/INTEGRATION_REPORT.md verdicts (READY first)

```
Onboard <REPO> into the AI Doctor using BACKEND_PLAN §3.2 and docs/INTEGRATION_REPORT.md.
1. Sandbox: load the original weights ONLY inside a throwaway Docker container with
   `--network none`, mounting backend/vendor/<model_id> read-only. Confirm the SHA-256 matches
   the registry. Reproduce one prediction with the repo's own preprocessing.
2. Convert: export to ONNX (skl2onnx / onnxmltools / tf2onnx / torch.onnx). Verify the ONNX
   output matches the original on 20 rows/images (max abs diff reported). If conversion is
   impossible, pin exact library versions in that worker and explain why.
3. Adapter: write workers/<tabular|torch|tf>/adapters/<model_id>.py: validate inputs
   (units, ranges), replicate preprocessing exactly, predict, explain (SHAP top-3 for tabular,
   Grad-CAM for images). Copy only the needed artefacts into workers/<...>/models/<model_id>/.
4. Registry: input_schema (JSON Schema; mark age, sex, bmi, height, weight as from_profile),
   output_labels with plain-language names, trained_on, limitations (population, dataset size),
   licence, and a retrieval description using words patients actually say.
5. Evaluate on a held-out public dataset for the same task (tell me which and its licence).
   Record accuracy, sensitivity, specificity and AUC. No dataset → metrics null, verified=false.
6. Add 5 phrases that SHOULD route to this model and 5 that should NOT to
   backend/tests/retrieval_cases.json. Run the retrieval tests.
7. Smoke test through the API and through the app (form or upload → result card).
45-minute rule applies. If it fails, set wave="backlog" with a reason and stop.
Stop and report: what worked, metrics, retrieval results, anything you're unsure about.
```

## Prompt 6: retrieval tuning and making it visible in the app

```
1. Run backend/tests/retrieval_cases.json across RETRIEVAL_THRESHOLD values 0.25–0.50 and show
   routing accuracy and false-positive rate for each. Pick the best value and set it in .env.example.
   If any model fires wrongly, improve its description (only the description).
2. Make the app's Model Library use GET /models from the backend instead of the bundled
   src/data/models.json when the backend is reachable, so integrated models show "Active" with
   their metrics and limitations. Keep the bundled file as an offline fallback.
3. In ModelResultCard, show the Grad-CAM/SHAP explanation when present.
4. Remove the offline-demo sample results for any model that is now really integrated
   (keep the demo only when the backend is unreachable, still labelled "Demo").
Run tsc, web export and backend tests. Stop and report.
```

## Prompt 7: full end-to-end check (before any demo)

```
Run a full end-to-end verification and write docs/VERIFICATION.md:
- docker compose up; all /health OK.
- For every integrated model: one typed conversation and one voice conversation that route to it,
  with screenshots from the web build at 390 px and 1366 px.
- 10 general questions that must NOT route to any model.
- The red-team safety set: every red-flag prompt escalates without calling the LLM.
- Latency for each route (p50/p90) and memory use of each worker.
List anything that failed and fix only clear bugs. Ask me before changing behaviour.
Stop and report.
```

---

## If something goes wrong (paste as needed)

**A repo won't load**
```
<REPO> fails to load. Show me the exact error. Check: library version used to save the model
(look at pickle protocol / sklearn version warnings / keras version), missing custom classes, and
whether the weights are complete. Try ONE fix (matching versions in a sandbox container). If it
still fails, apply the 45-minute rule: mark it backlog with the reason, and suggest whether
retraining on the same public dataset is a better path.
```

**Wrong model keeps getting picked**
```
The router picked <MODEL> for "<PHRASE>". Show the top-5 similarity scores for that phrase.
Fix it by improving model descriptions or the threshold only. Add the phrase to
retrieval_cases.json as a negative case and re-run the retrieval tests.
```

**The app can't reach the backend from my phone**
```
My phone shows "Offline demo". Check: backend runs with --host 0.0.0.0, EXPO_PUBLIC_API_URL in .env
uses my laptop's LAN IP (not localhost), the firewall allows port 8000, and Expo was restarted
after editing .env. Tell me the exact commands for my OS.
```
