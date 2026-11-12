# Environment and commands

Machine: Windows 11, 8 GB RAM (heavy work goes to Kaggle). Repo: `C:\Users\ANSHUMAN MOHANTY\Downloads\curanex-appfinalhtml\curanex-app`.
Python for the backend: `backend/.venv/Scripts/python` (3.11). Node 22 for the app.

## Start everything locally
- Docker Desktop must be running (`Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"`), then
  `docker start backend-worker-torch-1 backend-worker-tf-1` (image workers on ports 8102 and 8103; Postgres `curanex-postgres` too).
- In Claude Code desktop, use the preview launcher (`.claude/launch.json` in the parent folder) names:
  `curanex-worker-tabular` (port 8101), `curanex-api` (port 8001), `curanex-web` (Expo web, port 8081).
- Without the launcher: from `backend/`, `.venv/Scripts/python -m uvicorn workers.common.server:app --port 8101` and
  `.venv/Scripts/python -m uvicorn main:app --port 8001`; from the repo root `npx expo start --web --port 8081`.
- Health checks: `curl localhost:8001/health`, `localhost:8101/health`, `localhost:8102/health`, `localhost:8103/health`.
- After changing registry or tabular models: restart `curanex-worker-tabular` and `curanex-api`.
- After changing torch models/adapters: `docker cp` the files into `backend-worker-torch-1:/app/...`, `docker restart backend-worker-torch-1`,
  then `docker commit backend-worker-torch-1 backend-worker-torch:latest`.

## Tests and checks (run before saying a task is done)
- Backend: `cd backend && .venv/Scripts/python -m pytest -q` (105+ tests).
- App: `npx tsc --noEmit` and `npm run export:web`.

## Registry (which models are live)
- Regenerate after any model change, from `backend/`:
  `python tools/add_survey_models.py; python tools/add_image_models.py; python tools/apply_live_rule.py`
  (`apply_live_rule.py` holds RETIRED, DEMOTED and OUTSIDE_PASS: switched-off models, Unverified notes, outside passes.)
- Writes `backend/registry/model_registry.json` and `docs/model_registry.json` (kept identical).
- Regenerate `workflow/MODELS.md` with the small script in the git history of this file's commit, or by hand from the registry.

## Kaggle (free GPU/CPU)
- CLI: `python -m kaggle ...` with `KAGGLE_API_TOKEN` set in the command only. Ask the owner for the token; never save it.
- Account: `anshu004`. Notebook builders live in `kaggle/make_*_notebook.py`; each writes `kaggle/run_*/` (notebook + kernel-metadata.json).
- Push: `python -m kaggle kernels push -p kaggle/run_x`; status: `kernels status anshu004/<slug>`; results: `kernels output anshu004/<slug> -p kaggle/out_<slug>`.
- Max 2 GPU sessions at once; CPU kernels (enable_gpu false) do not count.
- The recipe that passes most often: NHANES lab-measured targets where the defining test is never an input, train 2017-18 -> test 2021-23 once
  (see `kaggle/run_nhanes_labs_*`). For survey (BRFSS) models: set the threshold on one NHANES cycle, test on another.

## Data and models on disk
- Tabular models: `backend/workers/tabular/models/{nhanes,survey,...}` (plain JSON numbers or boosted trees as JSON; no pickle at run time).
- Image models: `backend/workers/torch/models/multi/*.onnx|json` (SHA-256 pinned).
- Eval data: `backend/data/eval/` (git-ignored).
