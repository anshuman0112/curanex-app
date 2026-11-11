"""Round 4 Kaggle GPU notebooks (2026-11-11): TB chest X-ray and dermoscopy, each tested once on another hospital's data.
Run from the repo root: python kaggle/make_round4_notebooks.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_swap_notebooks import notebook  # noqa: E402  (importing also rewrites the earlier notebooks, unchanged)

TB_INTRO = """# Curanex: TB chest X-ray (GPU)

An earlier frozen-feature TB layer failed across sources (AUC 0.51-0.63). **Rules fixed before running (2026-11-11):**
- Train on TBX11K (China): TB against healthy and sick-but-not-TB; 15% stratified validation; whole network fine-tuned;
  threshold = highest with validation sensitivity >= 0.85.
- **Outside test, scored once:** NLM Shenzhen (China) and Montgomery County (USA) sets, TB against normal (file name ends _1 = TB).
- Verified if outside sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

TB_DATA = '''import pandas as pd
LABELS = ["TB", "No_TB"]
d = pd.read_csv(glob.glob("/kaggle/input/**/tbx11k-simplified/data.csv", recursive=True)[0]).drop_duplicates("fname")
img_dir = os.path.join(os.path.dirname(glob.glob("/kaggle/input/**/tbx11k-simplified/data.csv", recursive=True)[0]), "images")
rows = [(os.path.join(img_dir, r.fname), 0 if r.target == "tb" else 1) for r in d.itertuples() if os.path.exists(os.path.join(img_dir, r.fname))]
print("TBX11K training images:", len(rows), "TB:", sum(1 for _, y in rows if y == 0))
test = []
for f in glob.glob("/kaggle/input/**/images/images/*.png", recursive=True):
    name = os.path.basename(f)
    if name.startswith(("CHNCXR", "MCUCXR")):
        test.append((f, 1 if name.endswith("_1.png") else 0))
print("NLM test images:", len(test), "TB:", sum(y for _, y in test))'''

TB_RUN = '''net, predict, thr, best = train_model(rows, 2, epochs=10)
pt = predict(loader(test, eval_tf))[:, 0]
yt = np.array([y for _, y in test])
res = report(yt, pt, thr[0])
print("OUTSIDE TEST on NLM Shenzhen + Montgomery (once):", res)
export(net, "tb_xray", {"id": "tb_xray", "title": "TB chest X-ray", "labels": LABELS, "thresholds": thr, "validation_auc": round(best, 4),
                        "outside_test": res, "images": len(rows),
                        "trained_on": f"{len(rows)} TBX11K chest X-rays (China), full fine-tuning on a Kaggle GPU",
                        "outside_set": "NLM Shenzhen (China) and Montgomery County (USA), TB vs normal"})'''

DERM_INTRO = """# Curanex: dermoscopy lesion check (GPU)

The frozen-feature dermoscopy model failed on BCN20000 (sensitivity 0.49, AUC 0.697). **Rules fixed before running (2026-11-11):**
- Train on HAM10000 (Vienna, Queensland): suspicious = melanoma, basal cell carcinoma or actinic keratosis; validation = 15% of lesions;
  whole network fine-tuned; threshold = highest with validation sensitivity >= 0.85.
- **Outside test, scored once:** the same 1,500 BCN20000 lesions (Hospital Clinic Barcelona) used before; suspicious = malignant or actinic keratosis.
- Verified if outside sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

DERM_DATA = '''import pandas as pd
LABELS = ["suspicious", "not_suspicious"]
meta = pd.read_csv(glob.glob("/kaggle/input/**/HAM10000_metadata.csv", recursive=True)[0])
files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob("/kaggle/input/**/HAM10000_images_part_*/*.jpg", recursive=True)}
meta = meta[meta.image_id.isin(files)]
lesions = meta.lesion_id.unique()
rng = np.random.default_rng(42)
val_lesions = set(rng.choice(lesions, size=int(0.15 * len(lesions)), replace=False))
rows = [(files[r.image_id], 0 if r.dx in ("mel", "bcc", "akiec") else 1, r.lesion_id in val_lesions) for r in meta.itertuples()]
print("HAM10000 images:", len(rows), "suspicious:", sum(1 for r in rows if r[1] == 0))
picked = json.load(open(glob.glob("/kaggle/input/**/picked.json", recursive=True)[0]))
bcn_dir = os.path.dirname(glob.glob("/kaggle/input/**/picked.json", recursive=True)[0])
test = [(os.path.join(bcn_dir, p["id"] + ".jpg"), int(p["y"])) for p in picked if os.path.exists(os.path.join(bcn_dir, p["id"] + ".jpg"))]
print("BCN20000 test:", len(test), "suspicious:", sum(y for _, y in test))'''

DERM_RUN = '''# split by lesion (not image): train_model's random split is replaced here
import torch.nn as nn
train_rows = [(f, y) for f, y, v in rows if not v]
val_rows = [(f, y) for f, y, v in rows if v]
net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
net = net.to(DEV)
yt_ = np.array([y for _, y in train_rows]); counts = np.bincount(yt_, minlength=2)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(yt_) / (2 * counts), dtype=torch.float32, device=DEV))
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 12
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
dl_tr, dl_va = loader(train_rows, train_tf, True), loader(val_rows, eval_tf)
yv = np.array([y for _, y in val_rows])


def predict(dl):
    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.softmax(net(x.to(DEV)), 1).cpu().numpy())
    return np.concatenate(out)


best, state = -1, None
for ep in range(EPOCHS):
    net.train()
    for x, lab in dl_tr:
        opt.zero_grad(); loss = loss_fn(net(x.to(DEV)), lab.to(DEV)); loss.backward(); opt.step()
    sched.step()
    auc = roc_auc_score(yv == 0, predict(dl_va)[:, 0])
    print(f"epoch {ep + 1}: validation AUC {auc:.4f}")
    if auc > best:
        best, state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(state)
pv = predict(dl_va)
thr = []
for i in range(2):
    pos = pv[yv == i, i]
    thr.append(float(max(v for v in np.unique(pos) if (pos >= v).mean() >= 0.85)))
pt = predict(loader(test, eval_tf))[:, 0]
ytest = np.array([y for _, y in test])
res = report(ytest, pt, thr[0])
print("OUTSIDE TEST on BCN20000 (once):", res)
export(net, "dermoscopy_ft", {"id": "dermoscopy_ft", "title": "Dermoscopy lesion (fine-tuned)", "labels": LABELS, "thresholds": thr,
                              "validation_auc": round(best, 4), "outside_test": res, "images": len(train_rows),
                              "trained_on": f"{len(train_rows)} HAM10000 dermoscope images (Vienna, Queensland), split by lesion, full fine-tuning on a Kaggle GPU",
                              "outside_set": "BCN20000 (Hospital Clinic Barcelona), 1,500 lesions, one image per lesion"})'''

notebook("run_tb", TB_INTRO, [TB_DATA, TB_RUN],
         {"id": "anshu004/curanex-tb-xray", "title": "Curanex tb xray", "code_file": "tb_xray.ipynb",
          "dataset_sources": ["vbookshelf/tbx11k-simplified", "raddar/tuberculosis-chest-xrays-shenzhen", "raddar/tuberculosis-chest-xrays-montgomery"]})
notebook("run_derm", DERM_INTRO, [DERM_DATA, DERM_RUN],
         {"id": "anshu004/curanex-dermoscopy-ft", "title": "Curanex dermoscopy ft", "code_file": "dermoscopy_ft.ipynb",
          "dataset_sources": ["kmader/skin-cancer-mnist-ham10000", "anshu004/curanex-bcn-outside"]})
