"""Generated parts of docs/INTEGRATION_REPORT.md: the 126-repo summary table, the backlog and hospital-only
lines, and failed clones. Reads the registry + vendor/_clone_results.json; prints Markdown.

Usage (from backend/):  python tools/integration_tables.py > ../docs/_integration_tables.md
"""
import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REGISTRY = BACKEND.parent / "docs" / "model_registry.json"
RESULTS = BACKEND / "vendor" / "_clone_results.json"

# docs/TRAINING_PLAN.md: which track each backlog repo is planned for (repo name -> track)
TRACKS = {
    "sibisai/skin-scan": "Track 1: download weights (Hugging Face)",
    "deettoh/skin-lesion-classification": "Track 1: download weights (Hugging Face)",
    "sfu-mial/skin-IAV": "Track 1: download weights (Hugging Face / Zenodo)",
    "Whitedevil71/FundusNext": "Track 1: download weights (Hugging Face)",
    "li-xirong/mmc-amd": "Track 1: download weights (Google Drive)",
    "darwinai/TuberculosisNet": "Track 1: download weights (link in README)",
    "nyukat/breast_cancer_classifier": "Track 1: weights download script in repo",
    "nyukat/gmic": "Track 1: weights download script in repo",
    "vineet416/Chronic-Kidney-Disease-Prediction": "Track 2: retrain tabular (UCI CKD)",
    "yehonatanke/Thyroid-Predict-ML": "Track 2: retrain tabular (thyroid labs)",
    "SakH1075/Multiclass-Prediction-of-Thalassemia-Disorders-Using-Machine-Learning-Models-on-CBC-Parameters":
        "Track 2: retrain tabular (CBC)",
    "djparente/uti-ml": "Track 2: retrain tabular (urinalysis + symptoms)",
    "rick0110/cervical_cancer": "Track 2: retrain tabular (risk-factor questionnaire)",
    "PRAISE-Lab-Repository/PEPrML": "Track 2: retrain tabular (maternal clinical values)",
    "lanagarmire/PE_delivery": "Track 2: retrain tabular (maternal clinical values)",
    "Vankudoth-Sunil-Kumar/DiseasePrediction": "Track 2: retrain tabular (liver function tests)",
    "uxlabspk/NeuroVision": "Track 1 first (.tflite in repo), else Track 2 (questionnaire)",
    "ifyoungnet/MedGDM": "Track 2: retrain in Python (KNIME model file)",
    "resplab/accept": "Track 2: port the R model equations, or skip",
    "smon2006/Skin-Disease-Detection": "Track 3: fine-tune image model (Kaggle/Colab GPU)",
    "nawaz127/A-Multilayer-Classification-Framework-for-Skin-Disease-": "Track 3: fine-tune image model",
    "Duggineniakhil/oral-cancer-detection": "Track 3: fine-tune image model",
    "MarcoParola/oral_segmentation": "Track 3: fine-tune image model",
    "ShaonINT/Glaucoma_Detection": "Track 3: fine-tune image model",
    "anchor-neuro/retguard": "Track 3: fine-tune image model",
    "CASALab-Unisa/Dino2-DR": "Track 3: fine-tune image model",
    "pradamehavannan/OCT-Based-Retinal-Disease-Classification-and-AI-Clinical-Decision-Support":
        "Track 3: fine-tune image model",
    "sathvika-09/TB-detection": "Track 3: fine-tune image model",
    "Holy-Mahdi/ibids-xray-fracture-detection": "Track 3: fine-tune image model (YOLO)",
    "PingjunChen/GradingKneeOA": "Track 3: fine-tune image model",
    "m-aljasem/pediatric-radiology-ai": "Track 3: fine-tune image model",
    "Eeman1113/dental_study": "Track 4: defer (MedGemma LoRA needs a large GPU)",
    "Bhuvanaa28/Sentiment-Analysis-for-Depression-Detection": "Track 4: defer (mental-health text; support routing only)",
    "MarcoParola/detection_framework": "Track 4: defer (research framework, not one model)",
}

# docs/ALL_MODELS_PLAN.md: hospital-only repos -> route (matched on a distinctive part of the repo name)
ROUTES = [
    (("Seherin-Zaman", "BoranSekeroglu", "aramis-lab", "LNS98"), "Route A: scan upload, 2D (DICOM/slice)"),
    (("DeepIsles", "ISLES22_SEALS", "maynord", "uthmri", "ivadomed", "model-zoo", "R-Super", "PANORAMA",
      "SlicerPolycysticKidneySeg", "i2ireg", "FibUCL", "BIMCV-Prostate"), "Route A: scan upload, 3D (GPU worker)"),
    (("Pos4CHD", "Cardium", "czifan", "parkinson-datscan"), "Route E: clinic portal (specialist imaging)"),
    (("MSD-generalization", "mazder1/ECG", "Simsig"), "Route B: wearables / ECG export"),
    (("Parkinson_Detection_Using_Voice",), "Route C: phone voice recording"),
    (("cvpr2026-cv4chl",), "Route E: clinic portal (paediatric gait video)"),
    (("Dengue-Risk-Prediction", "Swasthya", "EpiSentinel"), "Route D: local health alerts (district weather/cases)"),
    (("vaibhav-k/Dengue",), "Route G: not feasible (virus genomics)"),
    (("CARDIAC-FM", "OPENEye-FM", "DB-EpiFM", "FetalCLIP"), "Route F: parked foundation model (needs fine-tuning + GPU)"),
    (("hears-lm", "PTSD-Methylation"), "Route G: not feasible (genetic lab data)"),
]
ROUTE_E_DEFAULT = "Route E: clinic portal (doctor uploads the clinical data)"


def route(repo: str) -> str:
    for keys, name in ROUTES:
        if any(k in repo for k in keys):
            return name
    return ROUTE_E_DEFAULT


def weights_cell(res: dict) -> str:
    w = res.get("weight_files") or []
    if not w:
        return "no"
    pointers = sum(1 for x in w if x.get("lfs_pointer"))
    return f"yes ({len(w)})" + (f", {pointers} LFS pointer" if pointers else "")


def main() -> None:
    models = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    results = {r["id"]: r for r in json.loads(RESULTS.read_text(encoding="utf-8"))}
    order = {"wave-1": 0, "wave-2": 1, "backlog": 2, "hospital-only": 3}
    models.sort(key=lambda m: (order.get(m["wave"], 9), m["id"]))

    print("## Summary of all 126 repos\n")
    print("| model_id | wave | tier | cloned | size (MB) | weights in repo | licence | frameworks |")
    print("|---|---|---|---|---|---|---|---|")
    for m in models:
        r = results.get(m["id"], {})
        status = {"ok": "yes", "failed": "FAILED", "needs_decision": "yes, >2 GB",
                  "metadata_only": "metadata only"}.get(r.get("status"), "not run")
        print(f"| {m['id']} | {m['wave']} | {m['app_tier']} | {status} | {r.get('size_mb', '')} | {weights_cell(r)} "
              f"| {r.get('license', '')} | {', '.join(r.get('frameworks') or []) or '?'} |")

    print("\n## Backlog repos (34): why not in the app yet, and what would make them usable\n")
    for m in models:
        if m["wave"] != "backlog":
            continue
        r = results.get(m["id"], {})
        why = "no model weights in the repo" if not r.get("weight_files") else "weights in repo but not usable as-is"
        links = sorted({e["host"] for e in r.get("external_downloads") or []})
        extra = f"; README links weights on {', '.join(links)}" if links else ""
        print(f"- `{m['id']}` ({', '.join(m['tasks'])}; input: {m['input']}): {why}{extra}. "
              f"Plan: {TRACKS.get(m['repo'], 'Track 2/3: retrain (see TRAINING_PLAN.md)')}.")

    print("\n## Hospital-only repos (69): why not in the phone app, and what would make them usable\n")
    for m in models:
        if m["wave"] != "hospital-only":
            continue
        r = results.get(m["id"], {})
        w = "weights in repo" if r.get("weight_files") else "no weights in repo"
        print(f"- `{m['id']}` ({', '.join(m['tasks'])}): needs {m['input']} that a phone user doesn't have; {w}. "
              f"Plan: {route(m['repo'])}.")

    failed = [r for r in results.values() if r.get("status") == "failed"]
    big = [r for r in results.values() if r.get("status") == "needs_decision"]
    print("\n## Clone problems\n")
    if not failed and not big:
        print("None: every repo cloned.")
    for r in failed:
        print(f"- FAILED `{r['id']}` ({r['url']}): {r.get('error', 'unknown error')}")
    for r in big:
        print(f"- OVER 2 GB after removing data folders `{r['id']}`: {r.get('size_mb')} MB "
              f"(removed: {', '.join(r.get('removed_dirs') or []) or 'nothing'}) -> needs your decision")
    for r in results.values():
        if r.get("status") == "metadata_only":
            print(f"- METADATA ONLY `{r['id']}`: cloned, but the files can't be checked out on Windows (paths with "
                  f"characters like ':'); file list, sizes, licence and weight hashes were read from Git objects instead.")
        elif r.get("skipped_paths"):
            print(f"- PARTIAL CHECKOUT `{r['id']}`: {len(r['skipped_paths'])} path(s) not valid on Windows were left out: "
                  f"{', '.join(r['skipped_paths'][:3])}")


if __name__ == "__main__":
    main()
