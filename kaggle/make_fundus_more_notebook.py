"""Kaggle GPU notebook (2026-11-12): pathologic myopia and AMD from a retina photo, trained on ODIR-5K (China), tested on RFMiD (India).
Run from the repo root: python kaggle/make_fundus_more_notebook.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_swap_notebooks import notebook  # noqa: E402

INTRO = """# Curanex: pathologic myopia and AMD from a retina photo (GPU)

**Rules fixed before running (2026-11-12):**
- Train on ODIR-5K (Peking University, China), one image per eye, preprocessed 512x512: multi-label, myopia ("M") and AMD ("A")
  against every other eye; 15% of patients for validation; whole EfficientNet-B0 fine-tuned (two sigmoid outputs).
- Thresholds: set once on the RFMiD *training* set (India; sensitivity >= 0.85 there). The ODIR-validation threshold is reported too.
- **Outside test, scored once:** RFMiD evaluation + test sets (other patients, never used): MYA and ARMD labels, each vs every other photo.
- Each finding is Verified if its sensitivity and AUC are both >= 0.80; live at >= 0.75; otherwise off."""

DATA = r'''import pandas as pd, ast
full = pd.read_csv(glob.glob("/kaggle/input/**/full_df.csv", recursive=True)[0])
img_dir = os.path.dirname(glob.glob("/kaggle/input/**/preprocessed_images/*.jpg", recursive=True)[0])
rows = []
for r in full.itertuples():
    f = os.path.join(img_dir, r.filename)
    if os.path.exists(f):
        lab = ast.literal_eval(r.labels) if isinstance(r.labels, str) else []
        rows.append((f, int("M" in lab), int("A" in lab), int(r.ID)))
rows = sorted(set(rows))
print("ODIR eyes:", len(rows), "myopia:", sum(r[1] for r in rows), "AMD:", sum(r[2] for r in rows))
def rfmid(split):
    csv = [c for c in glob.glob("/kaggle/input/**/RFMiD_*Labels.csv", recursive=True) if split in os.path.basename(c)][0]
    d = pd.read_csv(csv); folder = None
    for c in glob.glob("/kaggle/input/**/*.png", recursive=True):
        if split in c or (split == "Validation" and "/Validation/" in c) or (split == "Testing" and "/Test/" in c):
            folder = os.path.dirname(c); break
    out = []
    for r in d.itertuples():
        f = os.path.join(folder, f"{r.ID}.png")
        if os.path.exists(f):
            out.append((f, int(r.MYA), int(r.ARMD)))
    return out
cal = rfmid("Training")
test = rfmid("Validation") + rfmid("Testing")
print("RFMiD calibration:", len(cal), sum(r[1] for r in cal), sum(r[2] for r in cal), "| test:", len(test), sum(r[1] for r in test), sum(r[2] for r in test))
assert len(rows) > 5000 and len(cal) > 1500 and len(test) > 1000'''

RUN = r'''class MDS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return self.tf(Image.open(it[0]).convert("RGB")), torch.tensor([float(it[1]), float(it[2])])
mload = lambda items, tf, sh=False: torch.utils.data.DataLoader(MDS(items, tf), batch_size=32, shuffle=sh, num_workers=2)
pids = sorted({r[3] for r in rows}); rng = np.random.default_rng(42)
val_p = set(rng.choice(pids, size=int(0.15 * len(pids)), replace=False))
tr = [r for r in rows if r[3] not in val_p]; va = [r for r in rows if r[3] in val_p]
net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2); net = net.to(DEV)
pos = np.array([[r[1], r[2]] for r in tr]).mean(0)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32, device=DEV))
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); EPOCHS = 12
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
def pred(items):
    net.eval(); out = []
    with torch.no_grad():
        for x, _ in mload(items, eval_tf):
            out.append(torch.sigmoid(net(x.to(DEV))).cpu().numpy())
    return np.concatenate(out)
yv = np.array([[r[1], r[2]] for r in va]); best, state = -1, None
for ep in range(EPOCHS):
    net.train()
    for x, lab in mload(tr, train_tf, True):
        opt.zero_grad(); loss = loss_fn(net(x.to(DEV)), lab.to(DEV)); loss.backward(); opt.step()
    sched.step(); pv = pred(va)
    auc = np.mean([roc_auc_score(yv[:, k], pv[:, k]) for k in range(2)])
    print(f"epoch {ep + 1}: validation AUC {auc:.4f}", flush=True)
    if auc > best:
        best, state = auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(state); pv = pred(va)
thr_odir = [float(max(v for v in np.unique(pv[yv[:, k] == 1, k]) if (pv[yv[:, k] == 1, k] >= v).mean() >= 0.85)) for k in range(2)]
pc = pred(cal); yc = np.array([[r[1], r[2]] for r in cal])
thr = [float(max(v for v in np.unique(np.round(pc[yc[:, k] == 1, k], 6)) if (pc[yc[:, k] == 1, k] >= v).mean() >= 0.85)) for k in range(2)]
pt = pred(test); yt = np.array([[r[1], r[2]] for r in test])
res = {}
for k, name in enumerate(["myopia", "amd"]):
    res[name] = {"calibrated": report(yt[:, k], pt[:, k], thr[k]), "odir_threshold": report(yt[:, k], pt[:, k], thr_odir[k])}
    print(f"OUTSIDE TEST on RFMiD eval+test (once): {name}", res[name])
class Sig(nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n
    def forward(self, x):
        return torch.sigmoid(self.n(x))
full_net = Sig(net.cpu()).eval(); path = "/kaggle/working/fundus_more.onnx"
args = dict(input_names=["image"], output_names=["probs"], opset_version=17, dynamic_axes={"image": {0: "n"}})
try:
    torch.onnx.export(full_net, torch.randn(1, 3, 224, 224), path, dynamo=False, **args)
except TypeError:
    torch.onnx.export(full_net, torch.randn(1, 3, 224, 224), path, **args)
json.dump({"id": "fundus_more", "labels": ["myopia", "amd"], "thresholds": thr, "thresholds_odir": thr_odir, "validation_auc": round(best, 4),
           "outside_test": res, "mean": MEAN, "std": STD, "onnx_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
           "trained_on": f"{len(tr)} ODIR-5K retina photos (China), split by patient, full fine-tuning on a Kaggle GPU",
           "outside_set": "RFMiD (India): thresholds set on its training set, tested once on its evaluation and test sets"},
          open("/kaggle/working/fundus_more.json", "w"), indent=1)'''

notebook("run_fundus_more", INTRO, [DATA, RUN],
         {"id": "anshu004/curanex-fundus-more", "title": "Curanex fundus more", "code_file": "fundus_more.ipynb",
          "dataset_sources": ["andrewmvd/ocular-disease-recognition-odir5k", "andrewmvd/retinal-disease-classification"]})
print("written")
