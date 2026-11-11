"""Kaggle CPU notebook: outside test of the Parkinson's voice check on the Italian Parkinson's Voice and Speech recordings.
Run from the repo root: python kaggle/make_voice_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTRO = """# Curanex: voice check, outside test on Italian recordings (CPU)

**Rules fixed before running (2026-11-11):** the live voice model (trained on Istanbul studio recordings) is used unchanged.
Recordings: sustained vowel "a" (files starting VA1 / VA2). People with Parkinson's against the elderly healthy controls (age-matched;
young controls left out). Sex from the file code. One score per person (mean of their recordings); recordings the check refuses are skipped
and counted. Scored once. Verified if sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

CODE = [r'''import glob, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
src = os.path.dirname(glob.glob("/kaggle/input/**/voice_check.py", recursive=True)[0])
os.makedirs("/kaggle/working/api/voice_models", exist_ok=True)
import shutil
shutil.copy(os.path.join(src, "voice_check.py"), "/kaggle/working/api/voice_check.py")
for f in ("parkinson_voice.json", "parkinson_voice.onnx"):
    shutil.copy(os.path.join(src, f), f"/kaggle/working/api/voice_models/{f}")
open("/kaggle/working/api/__init__.py", "w").close()
sys.path.insert(0, "/kaggle/working")
from api import voice_check as v
print("available:", v.available())''',
r'''people = {}
for f in glob.glob("/kaggle/input/**/*.wav", recursive=True):
    if "/.cache/" in f:
        continue
    name = os.path.basename(f)
    if not name.upper().startswith(("VA1", "VA2")):
        continue
    parts = [q.lower() for q in f.split("/")[4:-1]]  # folders below the dataset root, not the dataset name
    if any("elderly healthy" in q for q in parts):
        y = 0
    elif any("people with parkinson" in q for q in parts):
        y = 1
    else:
        continue  # young controls are not age-matched
    if len(people) < 3: print("example path:", f)
    m = re.search(r"\d\d([MF])\d{8}", name)
    if not m:
        continue
    person = os.path.basename(os.path.dirname(f))
    people.setdefault((person, y), {"sex": "Male" if m.group(1) == "M" else "Female", "files": []})["files"].append(f)
print("people:", len(people), "with Parkinson's:", sum(1 for (_, y) in people if y == 1))
assert 0 < sum(1 for (_, y) in people if y == 1) < len(people), "both groups needed"
ys, scores, refused, flags = [], [], 0, []
for (person, y), info in sorted(people.items()):
    s, fl = [], []
    for f in info["files"]:
        try:
            r = v.check(open(f, "rb").read(), info["sex"])
            s.append(r["score"]); fl.append(r["flagged"])
        except v.VoiceCheckError as e:
            refused += 1
    if s:
        ys.append(y); scores.append(float(np.mean(s))); flags.append(float(np.mean(s)) >= json.load(open("/kaggle/working/api/voice_models/parkinson_voice.json"))["threshold"])
ys, scores, flags = map(np.array, (ys, scores, flags))
res = {"n_people": int(len(ys)), "with_parkinsons": int(ys.sum()), "recordings_refused": refused,
       "sensitivity": round(float(flags[ys == 1].mean()), 3), "specificity": round(float((~flags[ys == 0]).mean()), 3),
       "auc": round(float(roc_auc_score(ys, scores)), 3)}
low = min(res["sensitivity"], res["auc"]); res["status"] = "verified" if low >= 0.80 else "live" if low >= 0.75 else "off"
print("OUTSIDE TEST on Italian recordings (once):", res)
json.dump(res, open("/kaggle/working/voice_outside.json", "w"), indent=1)''']

cells = [{"cell_type": "markdown", "metadata": {}, "source": INTRO},
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "!pip install -q praat-parselmouth imageio-ffmpeg onnxruntime"}]
cells += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in CODE]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "run_voice"
d.mkdir(exist_ok=True)
(d / "voice_outside.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({"id": "anshu004/curanex-voice-outside", "title": "Curanex voice outside", "code_file": "voice_outside.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": ["ieeeai/italian-parkinsons-voice-and-speech", "anshu004/curanex-voice-model"], "competition_sources": [], "kernel_sources": []}, indent=1))
print("written", d)
