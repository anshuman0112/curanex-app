# Curanex AI Doctor: the complete workflow, step by step

This is the single source of truth for **how the 126 model repos become part of the AI Doctor** and **what happens, step by step, every time a user asks something**. It's written so you can hand it to Claude Code.

There are two workflows:

- **Part A: Build time (done once per repo).** Download → inspect → sort → sandbox → convert → wrap → describe → test → switch on.
- **Part B: Run time (every question).** App → safety → context → **model search** → orchestrator → worker → explanation → app.

Part C explains model search in detail. Part D shows how all the files chain together. Part E is the prompt for Claude Code.

---

# Part A: Build time (repo → working model inside the AI Doctor)

## A0. One-time setup

| Step | What | Command / file |
|---|---|---|
| A0.1 | Install Node 20+, Python 3.11, Git, Docker Desktop | |
| A0.2 | Unzip `curanex-app`, open a terminal in it | |
| A0.3 | Create `backend/.env` from `.env.example`; fill in `LLM_API_KEY` (and `OPENAI_API_KEY` for voice) | `backend/.env` |
| A0.4 | Install the app: `npm install` | |
| A0.5 | Create the backend environment: `cd backend && python -m venv .venv && pip install -r requirements/api.txt -r requirements/dev.txt` | |
| A0.6 | Check: `pytest -q` passes, `uvicorn main:app` → `/health` returns ok | |

## A1. Download all repos (three levels, so the disk doesn't fill up)

All 126 repos are listed in `docs/model_registry.json`. Each entry has a `wave`:

| Wave | Count | How it's downloaded | Why |
|---|---|---|---|
| `wave-1` | 8 | **Full clone**, latest commit only (`--depth 1`) | Integrated first |
| `wave-2` | 15 | **Full clone** (`--depth 1`) | Integrated second |
| `backlog` | 34 | **Metadata only**: file list and README, no big files (`--filter=blob:none --no-checkout`, then `git show HEAD:README.md`) | No usable weights yet. Kept so you can promote one later |
| `hospital-only` | 69 | **Metadata only**, same as backlog | Needs MRI/CT/EEG/ICU data. Shown in the Model Library, never run from the app |

Everything lands in `backend/vendor/<model_id>/`. That folder is in `.gitignore`, so it never goes to GitHub.

**Steps for each repo:**
1. **A1.1** Clone it using the level from the table above. Big files stored with Git LFS are skipped (`GIT_LFS_SKIP_SMUDGE=1`) unless they're the weights.
2. **A1.2** Record the commit SHA. This pins exactly which version you used.
3. **A1.3** List the files. Find the weights (`.pkl .joblib .sav .h5 .keras .pt .pth .onnx .tflite .safetensors`), the requirements file, the LICENSE, the training scripts and notebooks, and any data files.
4. **A1.4** For full clones: record each weight file's size and its SHA-256 fingerprint, which detects later tampering.
5. **A1.5** Note any external weight downloads mentioned in the README (Google Drive, Hugging Face, Zenodo, Kaggle).
6. **A1.6** Write all of this into the registry entry's `audit` fields.

**Never** run `pip install -r` from a repo on your own machine, and never run its scripts there.

## A2. Inspect (read the code, don't run it)

For each wave-1/2 repo, Claude Code reads the training script or notebook and writes this down:

| What to find | Example (diabetes) |
|---|---|
| The model file and its type | `models/final_model.joblib`, scikit-learn Pipeline (scaler + XGBoost) |
| The exact input features, **in order**, with units | Pregnancies, Glucose (mg/dL), BloodPressure (mmHg), SkinThickness (mm), Insulin (µU/mL), BMI, DiabetesPedigree, Age |
| Preprocessing | Zeros replaced with the median; StandardScaler inside the pipeline |
| Output classes | 0 = lower risk, 1 = higher risk |
| Training data | Pima Indians Diabetes (768 adult women) |
| Licence | None found → covered by the author's written permission (see `docs/permissions/`) |
| Framework and version | scikit-learn 1.x, xgboost 2.x (from requirements or pickle warnings) |

The output is one section per repo in `docs/INTEGRATION_REPORT.md`, ending with a verdict: **READY**, **NEEDS WORK** (what's missing) or **SKIP** (why).

## A3. Assign each model to a worker

| Weights format | Worker | Folder |
|---|---|---|
| `.pkl .joblib .sav` (sklearn / xgboost / lightgbm), `.onnx` tabular | **worker-tabular** | `backend/workers/tabular/` |
| `.pt .pth`, torchxrayvision, timm, YOLO (`ultralytics`) | **worker-torch** | `backend/workers/torch/` |
| `.h5 .keras .tflite` | **worker-tf** | `backend/workers/tf/` |

The reason for separate workers is that TensorFlow, PyTorch and different scikit-learn versions can't share one Python environment. Each worker has its own Docker image and requirements file.

## A4. Load in a sandbox (safety step)

Opening a pickle file runs code that's hidden inside it, so for every model:

1. **A4.1** Start a throwaway container, with no internet and the repo folder mounted read-only:
   `docker run --rm --network none -v $PWD/backend/vendor/<id>:/repo:ro curanex-worker-<type> python /tools/sandbox_load.py <id>`
2. **A4.2** Check that the weight file's SHA-256 matches the registry.
3. **A4.3** Load the model the way the repo does.
4. **A4.4** Run **one prediction** on a sample input, using the repo's own preprocessing.
5. **A4.5** Save that input and output as a "golden" test case: `tests/golden/<id>.json`.

## A5. Convert to a safe format (ONNX)

1. **A5.1** Convert with the matching tool:
   - scikit-learn → `skl2onnx`
   - XGBoost / LightGBM → `onnxmltools`
   - Keras → `tf2onnx`
   - PyTorch → `torch.onnx.export`
2. **A5.2** Run 20 test inputs through both the original and the ONNX version, and check the outputs match (max difference < 1e-4).
3. **A5.3** Save it as `backend/workers/<type>/models/<id>/model.onnx`, plus any scaler or encoder files.
4. **A5.4** If conversion is impossible, pin the exact library versions in that worker. Keep loading the original only inside that worker's container, never in the API.

## A6. Write the adapter (the "wrapper")

File: `backend/workers/<type>/adapters/<id>.py`

```python
class Adapter(ModelAdapter):
    id = "moizah1_diabetes_prediction"
    def load(self):        # open model.onnx once (lazy)
    def validate(self, x): # required fields present? units/ranges sane? -> list of problems
    def predict(self, x):  # same preprocessing as the repo -> ONNX -> Prediction
        return Prediction(model_id=self.id, label="Higher risk",
                          probabilities={"Lower risk": 0.31, "Higher risk": 0.69},
                          explanation={"top_factors": [["Glucose", +0.21], ["BMI", +0.09], ["Age", +0.05]]},
                          warnings=[])
```

Explanations:
- Tabular models: the 3 features that pushed the prediction most (SHAP).
- Image models: a Grad-CAM heatmap image showing where the model looked.

## A7. Describe it in the registry (this is what makes model search work)

Fill these fields in `backend/registry/model_registry.json`:

```jsonc
{
  "id": "moizah1_diabetes_prediction",
  "integrated": true, "verified": true, "worker": "tabular",
  "input_schema": { "type": "object", "required": ["glucose", "bmi", "age"],
    "properties": {
      "glucose": {"type":"number","unit":"mg/dL","minimum":40,"maximum":400,"label":"Fasting glucose"},
      "bmi":     {"type":"number","from_profile":true},
      "age":     {"type":"integer","from_profile":true},
      "bp":      {"type":"number","unit":"mmHg","label":"Diastolic BP","optional":true} } },
  "output_labels": ["Lower risk", "Higher risk"],
  "trained_on": "Pima Indians Diabetes dataset (768 adult women, USA)",
  "limitations": "Small, non-Indian population; women only; not a diagnosis. HbA1c test confirms.",
  "metrics": {"accuracy": 0.77, "sensitivity": 0.71, "specificity": 0.80, "auc": 0.83, "eval_set": "..."},
  "license": "none found",
  "permission": "Written permission from author, 2026-xx-xx (docs/permissions/moizah1_diabetes_prediction.pdf)",
  "description": "Type 2 diabetes risk check. For people who feel very thirsty, urinate often, feel tired, have high sugar readings, a family history of diabetes, or want to know their sugar risk. Uses fasting glucose, BMI and age."
}
```

`description` is written in **the words a patient would actually use**. Model search compares the user's message against this text, so a description that only says "Diabetes classifier" would almost never be found.

`metrics` must be **measured by you** (step A8), never copied from a README.

## A8. Measure accuracy

1. **A8.1** Pick a public labelled dataset for the same task, one the model **didn't** train on if possible. Record its name and licence.
2. **A8.2** Run every row or image through the adapter.
3. **A8.3** Calculate accuracy, sensitivity (catches real cases), specificity (avoids false alarms) and AUC.
4. **A8.4** Write the results into `metrics`. No dataset means `metrics: null` and `verified: false`. The model can still run, but the app labels it "Unverified".

## A9. Test model search for this model

Add 10 lines to `backend/tests/retrieval_cases.json`:

```json
{"text": "I'm always thirsty and pee a lot", "expect": "moizah1_diabetes_prediction"},
{"text": "my sugar reading was 160", "expect": "moizah1_diabetes_prediction"},
{"text": "what's a healthy breakfast", "expect": null},
{"text": "I have a rash on my arm", "expect": "omerkurtulus_skin_disease_classification"}
```

Run `pytest tests/test_retrieval.py`. The target is ≥ 85% correct, with few wrong models fired.

## A10. Switch it on

1. **A10.1** Set `integrated: true` (and `verified: true` if A8 passed).
2. **A10.2** Rebuild the embeddings: `python tools/embed_registry.py`.
3. **A10.3** Restart that worker: `docker compose up -d --build worker-<type>`.
4. **A10.4** Smoke test through the API: `POST /chat {"message": "I'm always thirsty"}` should return a `form_request` for this model.
5. **A10.5** Smoke test in the app: type it, fill in the form, and you should get a result card and an explanation.
6. **A10.6** Commit: `git commit -m "Integrate <repo> (<task>)"`.

**45-minute rule:** if A4–A6 aren't working after about 45 minutes, set `wave: "backlog"` with a reason and move on to the next repo.

---

# Part B: Run time (what happens on every question)

Example: the user **says** "I've been really thirsty and tired lately".

| # | Where | What happens | Data passed on |
|---|---|---|---|
| B1 | App · Voice screen | User taps the mic, speaks, then taps again. The audio is recorded (`.m4a` on phone, `.webm` on web) | audio file |
| B2 | App → `POST /transcribe` | Upload the audio | `multipart file` |
| B3 | Backend · STT | Whisper turns the speech into text (English or Hindi) | `{"text": "I've been really thirsty and tired lately"}` |
| B4 | App → `POST /chat` | Send the text, the user context, the conversation history, and `via: "voice"` | see JSON below |
| B5 | Backend · Safety pre-check | Deterministic red-flag rules. **If one matches → return `escalate` immediately** (no LLM, no model) | `escalate` block or continue |
| B6 | Backend · Context | Build a short user summary: age 34, male, BMI 27.1, hypertension, amlodipine, last meals, last workout | context text |
| B7 | Backend · **Model search** | Turn the message into an embedding, compare it with every integrated model's description, keep the top 4 above the threshold (Part C) | `[diabetes 0.58, thyroid 0.41, anemia 0.37]` |
| B8 | Backend · Orchestrator | Call the LLM with the context, the message, and **one tool per candidate model** plus `request_user_input` and `request_image` | LLM decides |
| B9 | LLM decision | (a) Answer directly; (b) a model fits but inputs are missing → `request_user_input`; (c) a model fits and the inputs are complete → call the model tool | |
| B10 | Backend → app | Case (b): reply with text plus a `form_request` (age and BMI pre-filled from the profile, glucose missing) | `form_request` block |
| B11 | App | Shows the form. In voice mode it also says: "I need a few details. I've opened a short form." | |
| B12 | App → `POST /chat` | User fills in glucose = 132. Sent with `form_values` and `model_id` | |
| B13 | Backend · Orchestrator | Validates the inputs against `input_schema` (units, ranges) | |
| B14 | Backend → worker | `POST worker-tabular/predict/moizah1_diabetes_prediction` | `{"glucose":132,"bmi":27.1,"age":34}` |
| B15 | Worker · Adapter | `validate` → same preprocessing as the repo → ONNX run → SHAP top factors | `Prediction` |
| B16 | Worker → backend | Returns the result | `{"label":"Higher risk","probabilities":{...},"explanation":{...}}` |
| B17 | Backend · Orchestrator | Gives the result to the LLM: "explain this for the user; not a diagnosis; say which test confirms it (HbA1c); voice → 2–4 sentences" | |
| B18 | Backend · Safety post-check | Removes any doses, blocks "you have X" wording, makes sure a "see a doctor" line is present | |
| B19 | Backend · Audit | Logs the model ID, version and latency (with an inputs hash, not the raw values) | `model_runs` row |
| B20 | Backend → app | `model_result` block plus `text` explanation block | blocks |
| B21 | App | Shows the result card (label, %, "trained on", limitations, "not a diagnosis") and speaks the summary aloud because it came in by voice | |
| B22 | App · Shared store | The conversation is saved, so switching to typing continues the same consult | |

**Example `/chat` request (B4):**
```json
{
  "session_id": "a1b2c3",
  "message": "I've been really thirsty and tired lately",
  "via": "voice",
  "context": {"profile": {"age": "34", "sex": "Male", "bmi": 27.1, "conditions": ["Hypertension"],
              "medications": ["Amlodipine 5 mg"], "allergies": ["Penicillin"], "diet": "Vegetarian"},
              "recent_meals": [{"name": "Poha", "kcal": 320}], "last_workout": null, "water_ml_today": 750},
  "history": [],
  "form_values": null, "image_id": null, "model_id": null
}
```

**Example reply (B20):**
```json
{"blocks": [
  {"type": "model_result", "model_id": "moizah1_diabetes_prediction", "model_name": "Type 2 diabetes risk",
   "label": "Higher risk", "probabilities": {"Lower risk": 0.31, "Higher risk": 0.69},
   "trained_on": "Pima Indians Diabetes dataset", "limitations": "Small, non-Indian population; not a diagnosis."},
  {"type": "text", "text": "Your answers suggest a higher chance of raised blood sugar. This is a screening estimate, not a diagnosis. An HbA1c blood test can confirm it. Please book one with your doctor this week."}
]}
```

**Image path (for example a chest X-ray):** same as above, except at B9 the LLM calls `request_image`. The app shows the upload card, the user picks a photo, and the app sends it with `POST /upload` → `image_id`. Then `/chat` goes out with `image_id` + `model_id`, the backend sends the image bytes to **worker-torch**, and the result comes back with a Grad-CAM heatmap.

**When no model fits** ("what's a healthy breakfast?"): at B7 nothing scores above the threshold. The LLM answers directly using the context (vegetarian, hypertension → low-salt options), and B13–B17 are skipped.

**Emergency** ("crushing chest pain spreading to my left arm"): B5 matches a red flag and returns `escalate` in under 300 ms. The app shows the red emergency card with Call 112 and Alert contacts, and speaks it. Nothing else runs. The app also runs this check itself, so it works with no internet.

---

# Part C: What "model search" means, exactly

**The problem:** you'll have dozens of models, and the user just talks normally. Something has to decide *which model, if any* fits what they said.

**The method (semantic search over model descriptions):**

1. **C1. Before any user arrives** (build time, step A10.2): each integrated model's `description` is turned into an **embedding**, a list of 384 numbers that captures its meaning, using the small local model `all-MiniLM-L6-v2`. These are saved to `backend/registry/embeddings.npy`.
2. **C2. When a message arrives** (B7): the message, plus the last 2 turns, is turned into an embedding the same way.
3. **C3. Compare:** compute the **cosine similarity** between the message and every model. The scale runs from 0 (unrelated) to 1 (same meaning).
4. **C4. Filter:** keep the **top 4** that score **at least 0.35** (`RETRIEVAL_THRESHOLD`; tuned in tests).
5. **C5. Hand over:** only those candidates are offered to the LLM as tools. The LLM still decides whether to use one, so a borderline match can be ignored.

**Example scores** for "I've been really thirsty and tired lately":

| Model | Description is about | Score | Offered? |
|---|---|---|---|
| Type 2 diabetes risk | thirst, frequent urination, tiredness, sugar | **0.58** | ✅ |
| Thyroid disease | tiredness, weight change, feeling cold | **0.41** | ✅ |
| Anemia | tiredness, pale skin, breathlessness | **0.37** | ✅ |
| Chest X-ray screening | X-ray image, cough, pneumonia | 0.12 | ❌ |
| Dental caries | tooth pain, cavities | 0.05 | ❌ |

The LLM sees three candidates, picks diabetes as the most relevant, and might mention that tiredness can also have other causes.

**Why not just a keyword list?** People say "always thirsty", "sugar high", "peeing at night" or "pyaas bahut lagti hai". Embeddings match the *meaning*, so they catch phrasings no keyword list would.

**Why not let the LLM see all 126 models?** It would be slow and expensive, and it would make more mistakes. Search narrows the choice to a few sensible options first.

**Why so few models in search?** Only models with `integrated: true` are searched. Hospital-only and backlog models are never offered, so the app can never ask a phone user for an MRI.

---

# Part D: How everything chains together (file map)

```
APP (Expo)
 src/app/(tabs)/doctor.tsx      text chat UI ─┐
 src/app/voice.tsx              voice UI ─────┤ both use
 src/store/doctor.tsx           shared conversation, send(), speak replies
 src/lib/api.ts                 chat() /transcribe /upload  (+ offline demo)
 src/lib/safety.ts              on-device red flags (works offline)
 src/lib/speech.ts              reads replies aloud (en-IN / hi-IN)
 src/components/blocks.tsx      cards: text, emergency, form, upload, model result
        │  HTTPS (EXPO_PUBLIC_API_URL)
        ▼
BACKEND API (FastAPI, after Prompt 1)
 api/routers/chat.py            POST /chat ─────────────► orchestrator/loop.py
 api/routers/voice.py           POST /transcribe ───────► clients/stt.py (Whisper)
 api/routers/uploads.py         POST /upload  (type/size check, EXIF strip)
 api/routers/models.py          GET /models   (Model Library)
 api/safety/rules.py            B5 red flags           api/safety/postcheck.py  B18
 api/orchestrator/context.py    B6 user summary
 api/orchestrator/retriever.py  B7 model search  ◄── registry/model_registry.json + embeddings.npy
 api/orchestrator/tools.py      turns candidate models into LLM tools (from input_schema)
 api/orchestrator/loop.py       B8–B17 tool loop (max 3 calls)
 api/clients/llm.py             LLM provider          api/clients/workers.py  HTTP to workers
        │  HTTP (WORKER_*_URL)
        ▼
MODEL WORKERS (Docker, one per framework)
 workers/common/server.py       /health  /models  /predict/{id}  (lazy load, LRU, 10 s timeout)
 workers/tabular/adapters/*.py  + models/<id>/model.onnx
 workers/torch/adapters/*.py    + models/<id>/...
 workers/tf/adapters/*.py       + models/<id>/...

BUILD-TIME TOOLS
 tools/clone_repos.py           A1  download by wave
 tools/probe_repos.py           A1.3–A1.6  audit fields
 tools/sandbox_load.py          A4  load inside a container with no network
 tools/convert_onnx.py          A5
 tools/evaluate.py              A8  metrics
 tools/embed_registry.py        A10.2  embeddings for model search
 vendor/<id>/                   the downloaded repos (never imported)
```

---

# Part E: The prompt for Claude Code

Paste **Prompt 0** from `docs/CLAUDE_CODE_PROMPTS.md` first, then Prompts 1 and 2. To download every repo, use this in place of Prompt 3:

```
Follow docs/WORKFLOW.md Part A, steps A1–A2, for ALL 126 repos in docs/model_registry.json.
Write tools/clone_repos.py and run it:
- wave-1 and wave-2: `git clone --depth 1` into backend/vendor/<model_id> (GIT_LFS_SKIP_SMUDGE=1,
  then fetch LFS files only if they are the weights).
- backlog and hospital-only: metadata only (`git clone --depth 1 --filter=blob:none --no-checkout`,
  then read README and LICENSE with `git show`). Do not check out large files.
- Record the commit SHA, file list, weight files, sizes and SHA-256 (full clones), licence and external
  weight links into each registry entry's `audit` field. Never run repo code or install its requirements.
- Report disk space used. If any single repo is over 2 GB, stop and ask me before keeping it.
Then do step A2 for the 23 wave-1/2 repos and write docs/INTEGRATION_REPORT.md with a verdict per repo.
Stop and report the verdict table.
```

Then, for each repo with a READY verdict:

```
Integrate <REPO> by following docs/WORKFLOW.md Part A steps A3 → A10 exactly, in order.
After each step, write one line in the report saying what was done and the result.
Apply the 45-minute rule. Never mark verified without measured metrics. Stop and report.
```

Finally, run Prompts 6 and 7 from `docs/CLAUDE_CODE_PROMPTS.md` (tuning, then the end-to-end check).
