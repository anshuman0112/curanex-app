"""Kaggle GPU notebook (2026-11-12): brain bleeding on CT, trained on the Turkish stroke CT set (Teknofest 2021), tested once on
PhysioNet CT-ICH (Iraq). Run from the repo root: python kaggle/make_ich_notebook.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_swap_notebooks import notebook  # noqa: E402

INTRO = """# Curanex: brain bleeding (intracranial haemorrhage) on CT (GPU)

**Rules fixed before running (2026-11-12):**
- Train on the Brain Stroke CT dataset (Turkey, Ministry of Health / Teknofest 2021): bleeding vs normal slices (ischaemia left out);
  15% stratified validation; whole EfficientNet-B0 fine-tuned; threshold = highest with validation sensitivity >= 0.85.
- **Outside test, scored once:** PhysioNet CT-ICH (Hssayeni et al., Iraq): every brain-window slice, any haemorrhage vs none
  (radiologist slice labels). Patient level (any slice flagged vs any slice bleeding) is reported too.
- Verified if slice-level sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

DATA = r'''import pandas as pd
LABELS = ["bleeding", "no_bleeding"]
rows = []
for f in glob.glob("/kaggle/input/**/Brain_Stroke_CT_Dataset/**/*", recursive=True):
    if not f.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    low = f.lower()
    if "/bleeding/" in low:
        rows.append((f, 0))
    elif "/normal/" in low:
        rows.append((f, 1))
rows = sorted(set(rows))
print("Turkish stroke CT:", len(rows), "bleeding:", sum(1 for _, y in rows if y == 0))
csv = glob.glob("/kaggle/input/**/hemorrhage_diagnosis*.csv", recursive=True)[0]
lab = pd.read_csv(csv)
root = os.path.dirname(csv)
test = []
for r in lab.itertuples():
    f = os.path.join(root, "Patients_CT", f"{int(r.PatientNumber):03d}", "brain", f"{int(r.SliceNumber)}.jpg")
    if os.path.exists(f):
        test.append((f, int(r.No_Hemorrhage == 0), int(r.PatientNumber)))
print("CT-ICH slices:", len(test), "with bleeding:", sum(t[1] for t in test), "patients:", len({t[2] for t in test}))
assert len(rows) > 1000 and len(test) > 1000'''

RUN = r'''net, predict, thr, best = train_model(rows, 2, epochs=10)
ps = predict(loader([(f, y) for f, y, _ in test], eval_tf))[:, 0]
ys = np.array([y for _, y, _ in test])
res = {"slice": report(ys, ps, thr[0])}
pat = {}
for (f, y, p), s in zip(test, ps):
    a = pat.setdefault(p, [0, 0.0]); a[0] = max(a[0], y); a[1] = max(a[1], float(s))
res["patient"] = report(np.array([v[0] for v in pat.values()]), np.array([v[1] for v in pat.values()]), thr[0])
print("OUTSIDE TEST on CT-ICH (once):", res)
export(net, "ich_ct", {"id": "ich_ct", "title": "Brain bleeding on CT", "labels": LABELS, "thresholds": thr, "validation_auc": round(best, 4),
                       "outside_test": res, "images": len(rows),
                       "trained_on": f"{len(rows)} brain CT slices from the Turkish Ministry of Health stroke set (bleeding vs normal), full fine-tuning on a Kaggle GPU",
                       "outside_set": "PhysioNet CT-ICH (Iraq), 82 patients, radiologist slice labels"})'''

notebook("run_ich", INTRO, [DATA, RUN],
         {"id": "anshu004/curanex-ich-ct", "title": "Curanex ich ct", "code_file": "ich_ct.ipynb",
          "dataset_sources": ["ozguraslank/brain-stroke-ct-dataset", "vbookshelf/computed-tomography-ct-images"]})
print("written")
