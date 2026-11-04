"""WORKFLOW.md A8 for mlmed_torchxrayvision: measure accuracy on the Kermany et al. test split (Mendeley Data
rscbjbr9sj v2, CC BY 4.0, 624 paediatric images). The dataset is not among the model's training sets.

Primary metric, fixed before running: the model's Pneumonia score vs the PNEUMONIA/NORMAL folder label, flagged
at the operating point 0.5. Run inside the worker image with no network:

  docker run --rm --network none -v <backend>/data/eval/chest_xray/test:/data:ro -v <backend>/workers:/app/workers:ro \
      -v <backend>/tools:/tools:ro backend-worker-torch python /tools/evaluate_xrv.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from workers.torch.adapters.mlmed_torchxrayvision import Adapter, preprocess

DATA = Path("/data")
SECONDARY = ["Pneumonia", "Consolidation", "Infiltration", "Lung Opacity"]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks of ties
        ranks[s == v] = ranks[s == v].mean()
    pos, neg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def report(y: np.ndarray, s: np.ndarray) -> dict:
    p = s >= 0.5
    tp, fn, tn, fp = (p & (y == 1)).sum(), (~p & (y == 1)).sum(), (~p & (y == 0)).sum(), (p & (y == 0)).sum()
    return {"n": int(len(y)), "positives": int(y.sum()), "accuracy": round(float((tp + tn) / len(y)), 3),
            "sensitivity": round(float(tp / (tp + fn)), 3), "specificity": round(float(tn / (tn + fp)), 3),
            "auc": round(auc(y, s), 3), "confusion": {"tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp)}}


def main() -> None:
    adapter = Adapter()
    adapter.load()
    idx = [adapter.labels.index(k) for k in SECONDARY]
    rows, labels = [], []
    for folder, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
        for path in sorted((DATA / folder).glob("*.jpeg")):
            scores, _ = adapter.scores(preprocess(Image.open(path)))
            rows.append(scores[0, idx])
            labels.append(label)
    s, y = np.array(rows), np.array(labels)
    print(json.dumps({"eval_set": "Kermany 2018 test split (paediatric, Guangzhou; CC BY 4.0)",
                      "primary_pneumonia_score": report(y, s[:, 0]),
                      "secondary_any_of": {"findings": SECONDARY, **report(y, s.max(axis=1))}}, indent=1))


if __name__ == "__main__":
    main()
