"""Kaggle GPU notebook (2026-11-12): breast cancer in biopsy slides, trained on BreakHis (Brazil), tested once on BACH (Portugal).
Replaces the Wisconsin needle-aspirate check, which has no outside data. Run from the repo root: python kaggle/make_breast_histo_notebook.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_swap_notebooks import notebook  # noqa: E402

INTRO = """# Curanex: breast cancer in biopsy slides (GPU)

**Rules fixed before running (2026-11-12):**
- Train on BreakHis (P&D Lab, Parana, Brazil), 200x magnification only: malignant vs benign; validation = 15% of patients
  (split by patient, not image); whole EfficientNet-B0 fine-tuned; best epoch by validation AUC; threshold = highest with
  validation sensitivity >= 0.85.
- **Outside test, scored once:** BACH / ICIAR 2018 training photos (Portugal, 200x, 400 labelled images): carcinoma in situ or
  invasive vs benign or normal. Each 2048x1536 photo is cut into a 3x3 grid (tiles close to BreakHis' field of view); photo score =
  mean tile score.
- Verified if outside sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

DATA = r'''import re
LABELS = ["malignant", "benign"]
rows = []
for f in glob.glob("/kaggle/input/**/BreaKHis_v1/**/200X/*.png", recursive=True):
    m = re.search(r"SOB_([BM])_[A-Z]+-(\d+-\d+[A-Z]*)-200-", os.path.basename(f))
    if m:
        rows.append((f, 0 if m.group(1) == "M" else 1, m.group(2)))
rows = sorted(set(rows))
patients = sorted({p for _, _, p in rows})
rng = np.random.default_rng(42)
val_p = set(rng.choice(patients, size=max(1, int(0.15 * len(patients))), replace=False))
print("BreakHis 200x:", len(rows), "malignant:", sum(1 for r in rows if r[1] == 0), "patients:", len(patients), "val patients:", len(val_p))
test = []
for f in glob.glob("/kaggle/input/**/*.tif", recursive=True):
    folder = os.path.basename(os.path.dirname(f)).lower()
    if "test" in f.lower():
        continue
    if folder in ("insitu", "invasive", "benign", "normal"):
        test.append((f, 1 if folder in ("insitu", "invasive") else 0))
test = sorted(set(test))
print("BACH photos:", len(test), "carcinoma:", sum(y for _, y in test))
assert len(rows) > 1000 and len(test) >= 300'''

RUN = r'''train_rows = [(f, y) for f, y, p in rows if p not in val_p]
val_rows = [(f, y) for f, y, p in rows if p in val_p]
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


def predict_batch(x):
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(x.to(DEV)), 1).cpu().numpy()


def predict(dl):
    return np.concatenate([predict_batch(x) for x, _ in dl])


best, state = -1, None
for ep in range(EPOCHS):
    net.train()
    for x, lab in dl_tr:
        opt.zero_grad(); loss = loss_fn(net(x.to(DEV)), lab.to(DEV)); loss.backward(); opt.step()
    sched.step()
    auc = roc_auc_score(yv == 0, predict(dl_va)[:, 0])
    print(f"epoch {ep + 1}: validation AUC {auc:.4f}", flush=True)
    if auc > best:
        best, state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(state)
pv = predict(dl_va)
thr = []
for i in range(2):
    pos = pv[yv == i, i]
    thr.append(float(max(v for v in np.unique(pos) if (pos >= v).mean() >= 0.85)))
scores = []
for f, _ in test:
    im = Image.open(f).convert("RGB"); w, h = im.size
    tiles = [im.crop((c * w // 3, r * h // 3, (c + 1) * w // 3, (r + 1) * h // 3)) for r in range(3) for c in range(3)]
    scores.append(float(predict_batch(torch.stack([eval_tf(t) for t in tiles]))[:, 0].mean()))
res = report(np.array([y for _, y in test]), np.array(scores), thr[0])
print("OUTSIDE TEST on BACH (once):", res)
export(net, "breast_histo", {"id": "breast_histo", "title": "Breast cancer in biopsy slides", "labels": LABELS, "thresholds": thr,
                             "validation_auc": round(best, 4), "outside_test": res, "images": len(train_rows), "tiles": 3,
                             "trained_on": f"{len(train_rows)} BreakHis 200x biopsy images (Brazil), split by patient, full fine-tuning on a Kaggle GPU",
                             "outside_set": "BACH / ICIAR 2018 (Portugal): 400 photos, carcinoma (in situ or invasive) vs benign or normal"})'''

notebook("run_breast_histo", INTRO, [DATA, RUN],
         {"id": "anshu004/curanex-breast-histo", "title": "Curanex breast histo", "code_file": "breast_histo.ipynb",
          "dataset_sources": ["ambarish/breakhis", "doucetbenton/iciar2018"]})
print("written")
