"""Whole-app check against the RUNNING servers (API :8001, tabular :8101, torch :8102, tf :8103). Run before a demo.

  python tools/full_app_check.py            -> prints PASS/FAIL per feature, writes data/eval/full_app_check.json

Covers: health, safety escalation, general chat, symptom routing, every live model through the real app path
(form or upload -> /chat), upload-anything recognition (report, X-ray, ECG), report explanation, voice transcription,
voice check, area alerts, experimental models, accounts + records, Doctor Desk PIN, library chat and 3-stage workflow.
Test values are typical adult values; images are public test images (data/eval). Results are checks that the feature works,
not accuracy numbers.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKEND = Path(__file__).resolve().parents[1]
EVAL = BACKEND / "data" / "eval"
API = "http://localhost:8001"
results: list[dict] = []


def env(name: str) -> str:
    for line in (BACKEND / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


def record(area: str, name: str, ok: bool, detail: str, t0: float) -> None:
    results.append({"area": area, "name": name, "ok": ok, "detail": detail[:240], "seconds": round(time.time() - t0, 1)})
    print(f"{'PASS' if ok else 'FAIL'}  [{area}] {name}  ({time.time() - t0:.1f}s)  {detail[:160]}", flush=True)


def chat(body: dict) -> list[dict]:
    body = {"session_id": f"check-{time.time()}", **body}
    return httpx.post(f"{API}/chat", json=body, timeout=240).json().get("blocks", [])


def text_of(blocks: list[dict]) -> str:
    return " ".join((b.get("text") or b.get("content") or "") for b in blocks if b["type"] in ("text", "markdown"))


def first(folder: Path) -> Path:
    return next(f for f in sorted(folder.rglob("*")) if f.suffix.lower() in (".jpg", ".jpeg", ".png"))


def oct_image() -> Path:
    d = np.load(EVAL / "medmnist" / "octmnist.npz")
    out = Path(tempfile.gettempdir()) / "curanex_oct.png"
    Image.fromarray(d["test_images"][0]).resize((496, 496)).convert("RGB").save(out)
    return out


IMAGES = {
    "mlmed_torchxrayvision": lambda: EVAL / "covid-chestxray-dataset" / "images" / "streptococcus-pneumoniae-pneumonia-temporal-evolution-1-day2.jpg",
    "ghifar_khder_knee_osteoarthritis_classification": lambda: first(BACKEND / "vendor" / "ghifar_khder_knee_osteoarthritis_classification" / "Test-data"),
    "joystonmenezes_xray_vision": lambda: first(EVAL / "mura" / "MURA-v1.1" / "valid" / "XR_WRIST"),
    "brain_mri_check": lambda: EVAL / "brain" / "x" / "Testing" / "glioma" / "Te-gl_1.jpg",
    "skin_lesion_check": lambda: EVAL / "isic_bcn" / "ISIC_0053462.jpg",
    "andreygermanov_yolov8_caries_detector": lambda: first(EVAL / "dentalai" / "test"),
    "khaled_ghanem_ocular_disease_classification": lambda: first(EVAL / "cataract" / "dataset"),
    "pamixsun_glaucoma": lambda: first(EVAL / "cataract" / "dataset" / "2_glaucoma"),
    "diabetic_retinopathy_check": lambda: EVAL / "idrid" / "x" / "Imagenes" / "Imagenes" / "IDRiD_001.jpg",
    "macular_edema_check": lambda: EVAL / "idrid_dme" / "test" / "Macular_edema" / "IDRiD_005test.jpg",
    "amd_oct_check": oct_image,
    "dme_oct_check": oct_image,
    "breast_ultrasound_check": lambda: EVAL / "busi" / "x" / "Dataset_BUSI_with_GT" / "benign" / "benign (1).png",
    "endoscopy_check": lambda: first(EVAL / "hyperkvasir" / "images" / "polyps"),
    "ulcerative_colitis_check": lambda: first(EVAL / "hyperkvasir" / "images" / "uc-1-2"),
    "esophagitis_check": lambda: first(EVAL / "esophagitis" / "Esophagitis"),
    "malaria_cell_check": lambda: first(EVAL / "malaria" / "x" / "cell_images" / "cell_images" / "Parasitized"),
}
FEMALE_ONLY = ("gestational", "pregnan", "maternal", "pcos", "cervical", "breast", "iron")


def form_values(model: dict) -> dict:
    """Typical adult answers from the model's own form: Yes for yes/no history questions, mid-range numbers."""
    female = any(w in model["id"].lower() or w in model.get("name", "").lower() for w in FEMALE_ONLY)
    out = {}
    for key, spec in (model.get("input_schema") or {}).get("properties", {}).items():
        if key == "age":
            out[key] = 52
        elif key == "sex":
            out[key] = "Female" if female else "Male"
        elif "enum" in spec:
            out[key] = "Yes" if "Yes" in spec["enum"] else spec["enum"][min(1, len(spec["enum"]) - 1)]
        elif spec.get("type") in ("number", "integer"):
            lo, hi = spec.get("minimum", 0), spec.get("maximum", 100)
            v = lo + 0.35 * (hi - lo)
            out[key] = int(round(v)) if spec["type"] == "integer" else round(v, 2)
    return out


def check_models() -> None:
    models = httpx.get(f"{API}/models", timeout=30).json()
    models = models.get("models", models) if isinstance(models, dict) else models
    live = [m for m in models if m.get("integrated")]
    for m in live:
        t0 = time.time()
        try:
            if m.get("upload_kind"):
                path = IMAGES[m["id"]]()
                image_id = httpx.post(f"{API}/upload", files={"file": (path.name, path.read_bytes())}, timeout=120).json()["image_id"]
                blocks = chat({"message": "Please check this image.", "image_id": image_id, "model_id": m["id"]})
            else:
                blocks = chat({"message": "", "model_id": m["id"], "form_values": form_values(m)})
            res = [b for b in blocks if b["type"] == "model_result"]
            if res:
                r = res[0]
                ok = bool(r.get("label")) and bool(r.get("trained_on")) and bool(r.get("limitations")) and bool(text_of(blocks))
                record("model", f"{m['name']} ({'Verified' if m.get('verified') else 'Unverified'})", ok, f"{r['label']} | flag {r.get('flag')}", t0)
            else:
                record("model", m["name"], False, json.dumps([b.get("type") for b in blocks]) + " " + text_of(blocks), t0)
        except Exception as e:  # report and keep going
            record("model", m.get("name", m["id"]), False, f"{type(e).__name__}: {e}", t0)
        time.sleep(1.5)  # free LLM rate limit


def lab_report() -> bytes:
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
    im = Image.new("RGB", (1240, 900), "white")
    d = ImageDraw.Draw(im)
    rows = ["CITY DIAGNOSTIC LABORATORY", "TEST RESULT UNIT REFERENCE", "Haemoglobin 10.2 g/dL 13.0 - 17.0", "Fasting blood sugar 162 mg/dL 70 - 100",
            "Triglycerides 245 mg/dL < 150", "HDL cholesterol 34 mg/dL > 40", "SGPT (ALT) 68 U/L < 45", "SGOT (AST) 52 U/L < 40", "TSH 3.1 uIU/mL 0.4 - 4.0"]
    for i, r in enumerate(rows):
        d.text((60, 50 + i * 80), r, font=f, fill="black")
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def ecg() -> bytes:
    W, H = 1600, 600
    im = Image.new("RGB", (W, H), (255, 244, 244))
    d = ImageDraw.Draw(im)
    for x in range(0, W, 10):
        d.line([(x, 0), (x, H)], fill=(240, 150, 150) if x % 50 == 0 else (248, 205, 205))
    for y in range(0, H, 10):
        d.line([(0, y), (W, y)], fill=(240, 150, 150) if y % 50 == 0 else (248, 205, 205))
    t = np.linspace(0, 10, 4000)
    y = sum(1.2 * np.exp(-((t - b) / 0.012) ** 2) + 0.3 * np.exp(-((t - b - 0.28) / 0.06) ** 2) + 0.12 * np.exp(-((t - b + 0.18) / 0.03) ** 2) for b in np.arange(0.4, 10, 0.83))
    d.line([(int(a / 10 * W), int(320 - v * 150)) for a, v in zip(t, y)], fill="black", width=2)
    d.text((20, 10), "12-lead ECG  25 mm/s  10 mm/mV  Lead II", font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22), fill="black")
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def main() -> None:
    for port, name in ((8001, "API"), (8101, "tabular worker"), (8102, "image worker (torch)"), (8103, "image worker (tf)")):
        t0 = time.time()
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=10).json()
            record("servers", name, bool(r.get("ok")), json.dumps(r)[:120], t0)
        except Exception as e:
            record("servers", name, False, f"down: {type(e).__name__}", t0)

    t0 = time.time()
    b = chat({"message": "I have crushing chest pain spreading to my left arm and I am sweating"})
    record("safety", "Emergency words -> SOS escalation (no AI)", any(x["type"] == "escalate" for x in b), json.dumps(b)[:160], t0)

    t0 = time.time()
    b = chat({"message": "What foods have a lot of iron?"})
    record("chat", "General health question", len(text_of(b)) > 80, text_of(b)[:160], t0)

    t0 = time.time()
    b = chat({"message": "I feel thirsty all the time and pass urine many times a day"})
    kinds = [x["type"] for x in b]
    record("chat", "Symptoms -> suggests a matching check", "form_request" in kinds or "model_result" in kinds or "upload_request" in kinds,
           f"{kinds} {[x.get('title') for x in b if x['type'] == 'form_request']}", t0)

    t0 = time.time()
    b = chat({"message": "I have a mild headache since morning, what can I take?"})
    record("chat", "Mild symptom -> quick-relief advice", len(text_of(b)) > 80, text_of(b)[:160], t0)

    if "--skip-models" not in sys.argv:
        check_models()

    for name, data, want in (("Lab report photo", lab_report(), "report"), ("ECG image", ecg(), "ecg"),
                             ("Chest X-ray image", IMAGES["mlmed_torchxrayvision"]().read_bytes(), "chest x-ray")):
        t0 = time.time()
        got = httpx.post(f"{API}/v1/identify", files={"file": ("f.png", data)}, timeout=240).json()
        ok = got.get("route") == want or got.get("kind") == want
        record("upload anything", f"{name} recognised", ok, f"{got.get('route')} / {got.get('kind')} / model {got.get('model_id')}", t0)
        if name == "Lab report photo" and got.get("route") == "report":
            t0 = time.time()
            b = chat({"message": "Please explain my report.", "report_text": got["text"]})
            record("upload anything", "Report explained", "haemoglobin" in text_of(b).lower() or "hemoglobin" in text_of(b).lower(), text_of(b)[:160], t0)
        if name == "ECG image" and got.get("route") == "image":
            t0 = time.time()
            b = chat({"message": f"I uploaded a {got['kind']}. Curanex has no checked model for this kind of file, so an AI image reader looked at it and noted: "
                                 f"{'; '.join(got.get('findings') or [])}. Explain in simple words, say it is not a diagnosis, and which doctor to see."})
            record("upload anything", "ECG -> AI reading answered", len(text_of(b)) > 80, text_of(b)[:160], t0)
        time.sleep(2)

    t0 = time.time()
    wav = Path(tempfile.gettempdir()) / "curanex_voice.wav"
    script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "$s.SetOutputToWaveFile($env:VERIFY_OUT); $s.Speak($env:VERIFY_TEXT); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True,  # noqa: S603, S607 fixed local command
                   env={**os.environ, "VERIFY_OUT": str(wav), "VERIFY_TEXT": "I have had a fever and cough for three days"})
    r = httpx.post(f"{API}/transcribe", files={"file": ("voice.wav", wav.read_bytes(), "audio/wav")}, timeout=120)
    said = (r.json().get("text") or "") if r.status_code == 200 else r.text
    record("voice", "Speech -> text (Whisper)", "fever" in said.lower() and "cough" in said.lower(), said, t0)

    for name, path in (("Voice check info", "/voice-check"), ("Area health alerts", "/alerts?district=Khordha"), ("Experimental models list", "/experimental")):
        t0 = time.time()
        r = httpx.get(f"{API}{path}", timeout=30)
        record("other", name, r.status_code == 200, r.text[:140], t0)

    # accounts, records, Doctor Desk (local TEST accounts only)
    acc = {line.split(":")[0]: line.split()[1:] for line in (BACKEND / "data" / "test_accounts.txt").read_text().splitlines() if ": " in line}
    t0 = time.time()
    tok = {}
    for role in ("admin", "doctor", "patient"):
        email, _, pw = acc[role]
        r = httpx.post(f"{API}/auth/login", json={"email": email, "password": pw}, timeout=30)
        tok[role] = r.json().get("token") if r.status_code == 200 else None
    record("accounts", "Admin, doctor and patient log in", all(tok.values()), str({k: bool(v) for k, v in tok.items()}), t0)
    if all(tok.values()):
        P, D, A = ({"Authorization": f"Bearer {tok[k]}"} for k in ("patient", "doctor", "admin"))
        t0 = time.time()
        case = httpx.post(f"{API}/records/cases", headers=P, timeout=30, json={"title": "Full app check", "symptoms": "fever and cough for 3 days, breathless",
                                                                               "patient": {"age": 45, "sex": "male"}}).json()
        docs = httpx.get(f"{API}/records/doctors", headers=P, timeout=30).json()
        shared = httpx.post(f"{API}/records/cases/{case['id']}/share", headers=P, json={"doctor_id": docs[0]["id"]}, timeout=30).status_code == 200
        seen = httpx.get(f"{API}/doctor/cases/{case['id']}", headers=D, timeout=30).status_code == 200
        plan = httpx.post(f"{API}/doctor/cases/{case['id']}/plan", headers=D, timeout=30,
                          json={"diagnosis": "Check only", "tests": ["Chest X-ray"], "medicines": [], "advice": "Automated check, ignore."}).status_code == 200
        back = any(c["id"] == case["id"] and c["doctor_plans"] for c in httpx.get(f"{API}/records/cases", headers=P, timeout=30).json())
        httpx.delete(f"{API}/records/cases/{case['id']}/share/{docs[0]['id']}", headers=P, timeout=30)
        record("records", "Patient saves + shares case, doctor sends plan, patient sees it", shared and seen and plan and back,
               f"shared {shared} doctor sees {seen} plan {plan} patient sees {back}", t0)
        t0 = time.time()
        locked = httpx.get(f"{API}/admin/doctors", headers=P, timeout=30).status_code in (401, 403) and httpx.get(f"{API}/admin/doctors", headers=A, timeout=30).status_code == 200
        record("accounts", "Admin pages: admin yes, patient no", locked, "", t0)

    pin = env("DOCTOR_DEV_KEY")
    t0 = time.time()
    ok = httpx.get(f"{API}/doctor/check", headers={"X-Doctor-Key": pin}, timeout=30).status_code == 200 and httpx.get(f"{API}/doctor/check", timeout=30).status_code == 403
    record("doctor desk", "PIN opens it, no PIN refused", ok, "", t0)
    case = {"patient": {"age": 45, "sex": "male", "weight_kg": 70, "kidney_disease": False, "liver_disease": False, "allergies": [], "current_medicines": []},
            "symptoms": "fever for 3 days, a lot of cough with yellow sputum, difficulty breathing",
            "report": {"WBC": "15,800 (high)", "SpO2": "93%", "Chest X-ray": "right lower lobe consolidation"}}
    H = {"X-Doctor-Key": pin}
    for name, path, body, test in (
        ("Ask the AI (library chat)", "/doctor/ask", {"case": case, "question": "Which tests should I order first?", "history": []}, lambda j: j["answer"] or j["general"]),
        ("1. Suggest tests", "/doctor/tests", case, lambda j: j["tests"]),
        ("2. Likely diseases", "/doctor/differential", case, lambda j: j["supported"]),
        ("3. Treatment with label doses", "/doctor/treatment", {"case": case, "disease": "Community-acquired pneumonia"}, lambda j: j["plan"] or j["medicines"]),
    ):
        t0 = time.time()
        r = httpx.post(f"{API}{path}", headers=H, json=body, timeout=400)
        j = r.json() if r.status_code == 200 else {}
        ok = r.status_code == 200 and bool(test(j))
        detail = json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in j.items() if k in ("answer", "general", "tests", "supported", "also_consider", "plan", "medicines", "dropped")})
        record("doctor desk", name, ok, detail if r.status_code == 200 else f"HTTP {r.status_code}", t0)

    passed = sum(r["ok"] for r in results)
    print(f"\n{passed}/{len(results)} checks passed")
    (EVAL / "full_app_check.json").write_text(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"), "passed": passed, "total": len(results), "results": results}, indent=1),
                                              encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
