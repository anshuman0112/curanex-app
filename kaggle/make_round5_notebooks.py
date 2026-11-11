"""Round 5 Kaggle GPU notebooks (2026-11-11): cervical cells, colorectal pathology and retinal OCT, each tested once on another centre.
Run from the repo root: python kaggle/make_round5_notebooks.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_swap_notebooks import notebook  # noqa: E402

RULES = "Whole network fine-tuned; 15% stratified validation; best epoch by validation AUC; threshold = highest with validation sensitivity >= 0.85. Verified if outside sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."

CERV_INTRO = f"""# Curanex: abnormal cervical cells, Pap smear (GPU)

Frozen features failed SIPaKMeD -> Herlev (AUC 0.595). **Rules fixed before running (2026-11-11):** train on SIPaKMeD cropped cells (Greece):
abnormal = dyskeratotic or koilocytotic. **Outside test, scored once:** Herlev (Denmark): abnormal = dysplasia or carcinoma in situ;
segmentation images (-d) left out. {RULES}"""

CERV_DATA = '''LABELS = ["abnormal", "normal"]
rows = []
for f in glob.glob("/kaggle/input/**/*.bmp", recursive=True) + glob.glob("/kaggle/input/**/*.BMP", recursive=True):
    low = f.lower()
    if "cropped" in low and ("sipakmed" in low or "im_" in low):
        rows.append((f, 0 if ("dyskeratotic" in low or "koilocytotic" in low) else 1))
rows = sorted(set(rows))
print("SIPaKMeD cells:", len(rows), "abnormal:", sum(1 for _, y in rows if y == 0))
test = []
for f in glob.glob("/kaggle/input/**/*", recursive=True):
    low = f.lower()
    if not low.endswith(".bmp") or "cropped" in low or "-d." in os.path.basename(low) or "sipakmed" in low or "im_" in low:
        continue
    folder = os.path.basename(os.path.dirname(low))
    if any(k in folder for k in ("dysplas", "carcinoma", "normal")):
        test.append((f, 1 if ("dysplas" in folder or "carcinoma" in folder) else 0))
test = sorted(set(test))
print("Herlev cells:", len(test), "abnormal:", sum(y for _, y in test))'''

CERV_RUN = '''net, predict, thr, best = train_model(rows, 2, epochs=12)
pt = predict(loader(test, eval_tf))[:, 0]
yt = np.array([y for _, y in test])
res = report(yt, pt, thr[0])
print("OUTSIDE TEST on Herlev (once):", res)
export(net, "cervix_ft", {"id": "cervix_ft", "title": "Abnormal cervical cells", "labels": LABELS, "thresholds": thr, "validation_auc": round(best, 4),
                          "outside_test": res, "images": len(rows), "trained_on": f"{len(rows)} SIPaKMeD Pap smear cells (Greece), full fine-tuning on a Kaggle GPU",
                          "outside_set": "Herlev (Denmark), dysplasia or carcinoma in situ vs normal cells"})'''

CRC_INTRO = f"""# Curanex: colorectal cancer in pathology tiles (GPU)

Frozen features failed Kather 2016 -> CRC-VAL-HE-7K (tumour sensitivity 0.142). **Rules fixed before running (2026-11-11):** train on
Kather 2016 (Mannheim) tiles, tumour vs other tissue ("complex stroma" left out as mixed), with strong colour (stain) augmentation.
**Outside test, scored once:** CRC-VAL-HE-7K (NCT, Heidelberg; 50 other patients), tumour vs everything else. {RULES}"""

CRC_DATA = '''LABELS = ["tumour", "no_tumour"]
rows = []
for f in glob.glob("/kaggle/input/**/Kather_texture_2016_image_tiles_5000/**/*.tif", recursive=True):
    folder = os.path.basename(os.path.dirname(f)).upper()
    if "COMPLEX" not in folder:
        rows.append((f, 0 if "TUMOR" in folder else 1))
print("Kather 2016 tiles:", len(rows), "tumour:", sum(1 for _, y in rows if y == 0))
test = []
for f in glob.glob("/kaggle/input/**/CRC-VAL-HE-7K/**/*.tif", recursive=True):
    test.append((f, 1 if os.path.basename(os.path.dirname(f)).upper() == "TUM" else 0))
print("CRC-VAL-HE-7K tiles:", len(test), "tumour:", sum(y for _, y in test))'''

CRC_RUN = '''train_tf.transforms.insert(5, T.ColorJitter(hue=0.1, saturation=0.4))  # extra stain variation
net, predict, thr, best = train_model(rows, 2, epochs=12)
pt = predict(loader(test, eval_tf))[:, 0]
yt = np.array([y for _, y in test])
res = report(yt, pt, thr[0])
print("OUTSIDE TEST on CRC-VAL-HE-7K (once):", res)
export(net, "crc_ft", {"id": "crc_ft", "title": "Colorectal cancer tiles", "labels": LABELS, "thresholds": thr, "validation_auc": round(best, 4),
                       "outside_test": res, "images": len(rows), "trained_on": f"{len(rows)} Kather 2016 H&E tiles (Mannheim), full fine-tuning on a Kaggle GPU",
                       "outside_set": "CRC-VAL-HE-7K (NCT Heidelberg, 50 other patients)"})'''

OCT_INTRO = f"""# Curanex: retinal OCT check (GPU)

**Rules fixed before running (2026-11-11):** train on Kermany 2018 OCT scans: CNV, DME, drusen, normal (train folder only).
**Outside test, scored once:** OCTDL (Russia): DME vs normal (NO) and AMD vs normal, where the AMD score is CNV + drusen.
The OCT check is Verified only if both findings pass. {RULES}"""

OCT_DATA = '''LABELS = ["CNV", "DME", "DRUSEN", "NORMAL"]
rows = []
for f in glob.glob("/kaggle/input/**/OCT2017*/train/*/*.jpeg", recursive=True):
    lab = os.path.basename(os.path.dirname(f)).upper()
    if lab in LABELS:
        rows.append((f, LABELS.index(lab)))
rows = sorted(set(rows))[::3]  # every third scan keeps training within the GPU session (about 28,000 scans)
print("Kermany training scans:", len(rows))
test = []
for f in glob.glob("/kaggle/input/**/*", recursive=True):
    low = f.lower()
    if not low.endswith((".jpg", ".jpeg", ".png")) or "oct2017" in low or "kermany" in low:
        continue
    folder = os.path.basename(os.path.dirname(f)).upper()
    if folder in ("AMD", "DME", "NO"):
        test.append((f, folder))
print("OCTDL scans:", {k: sum(1 for _, y in test if y == k) for k in ("AMD", "DME", "NO")})'''

OCT_RUN = '''net, predict, thr, best = train_model(rows, 4, epochs=6)
po = predict(loader([(f, 0) for f, _ in test], eval_tf))
lab = np.array([y for _, y in test])
results = {}
for name, cond, score, t in (("DME", "DME", po[:, 1], thr[1]), ("AMD", "AMD", po[:, 0] + po[:, 2], min(thr[0], thr[2]))):
    keep = (lab == cond) | (lab == "NO")
    results[name] = report((lab[keep] == cond).astype(int), score[keep], t)
    print("OUTSIDE TEST on OCTDL (once):", name, results[name])
export(net, "oct_ft", {"id": "oct_ft", "title": "Retinal OCT", "labels": LABELS, "thresholds": thr, "validation_auc": round(best, 4),
                       "outside_test": results, "images": len(rows), "trained_on": f"{len(rows)} Kermany 2018 OCT scans, full fine-tuning on a Kaggle GPU",
                       "outside_set": "OCTDL (Russia): DME and AMD vs normal"})'''

notebook("run_cervix", CERV_INTRO, [CERV_DATA, CERV_RUN],
         {"id": "anshu004/curanex-cervix-ft", "title": "Curanex cervix ft", "code_file": "cervix_ft.ipynb",
          "dataset_sources": ["marinaeplissiti/sipakmed", "niladrishekharray/herlev-pap-smear-2005"]})
notebook("run_crc", CRC_INTRO, [CRC_DATA, CRC_RUN],
         {"id": "anshu004/curanex-crc-ft", "title": "Curanex crc ft", "code_file": "crc_ft.ipynb",
          "dataset_sources": ["kmader/colorectal-histology-mnist", "imrankhan77/crc-val-he-7k"]})
notebook("run_oct", OCT_INTRO, [OCT_DATA, OCT_RUN],
         {"id": "anshu004/curanex-oct-ft", "title": "Curanex oct ft", "code_file": "oct_ft.ipynb",
          "dataset_sources": ["paultimothymooney/kermany2018", "orvile/octdl-optical-coherence-tomography-dataset"]})
