"""Area dengue alert from weather only. Trained by the Curanex team on the weekly district dengue counts collected by
BlackCatDisha/EpiSentinel (Karnataka, 27 districts, 2017-2023). Run from backend/.

The repo's models also use last week's case counts, which no free live source provides, so this model uses only weather
that Open-Meteo gives for any place today. Weather for training comes from the same Open-Meteo (ERA5) archive, so the
live inputs are computed the same way as the training inputs.

Target (the repo's): next week's cases above that district's usual level (its 75th percentile).
Rules fixed before looking at the test years:
- features: mean temperature, mean humidity and total rain over the last 4 weeks and the 4 weeks before, plus season
- train on 2017-2021, test on 2022-2023, scored once
- logistic regression unless boosting wins by 0.02 AUC in year-by-year folds
- "high" threshold: highest with sensitivity >= 0.80 on the 2021 season (held out from a 2017-2020 fit)
"""
import csv
import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
CASES = BACKEND / "vendor" / "blackcatdisha_episentinel" / "random_forest_trained" / "model_ready_district_week_trainable.csv"
CACHE = BACKEND / "data" / "eval" / "dengue_weather"
OUT = BACKEND / "api" / "alert_models" / "dengue_weather.json"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
TARGET_SENSITIVITY = 0.80
# District headquarters (approximate), used only to fetch weather for training.
PLACES = {
    "Bagalkote": (16.18, 75.70), "Ballari": (15.14, 76.92), "Belagavi": (15.85, 74.50), "Bengaluru Rural": (13.28, 77.54),
    "Bengaluru Urban": (12.97, 77.59), "Bidar": (17.91, 77.52), "Chamarajanagara": (11.92, 76.94), "Chikkamagaluru": (13.32, 75.77),
    "Chitradurga": (14.23, 76.40), "Dakshina Kannada": (12.87, 74.84), "Davangere": (14.46, 75.92), "Dharwad": (15.46, 75.01),
    "Gadag": (15.43, 75.63), "Hassan": (13.00, 76.10), "Haveri": (14.79, 75.40), "Kalaburagi": (17.33, 76.83), "Kodagu": (12.42, 75.74),
    "Kolar": (13.14, 78.13), "Koppal": (15.35, 76.15), "Mandya": (12.52, 76.90), "Mysuru": (12.30, 76.64), "Raichur": (16.20, 77.36),
    "Shivamogga": (13.93, 75.57), "Tumakuru": (13.34, 77.10), "Udupi": (13.34, 74.75), "Uttara Kannada": (14.80, 74.13), "Vijayapura": (16.83, 75.71),
}
FEATURES = ["temp_recent", "temp_before", "humidity_recent", "humidity_before", "rain_recent", "rain_before", "week_sin", "week_cos"]


def weather(name: str) -> dict[str, tuple[float, float, float]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name.replace(' ', '_')}.json"
    if not path.exists():
        lat, lon = PLACES[name]
        for attempt in range(5):
            r = httpx.get(ARCHIVE, params={"latitude": lat, "longitude": lon, "start_date": "2016-10-01", "end_date": "2023-12-31",
                                           "daily": DAILY, "timezone": "Asia/Kolkata"}, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        path.write_text(r.text)
    d = json.loads(path.read_text())["daily"]
    return {t: (a, b, c or 0.0) for t, a, b, c in zip(d["time"], d["temperature_2m_mean"], d["relative_humidity_2m_mean"], d["precipitation_sum"], strict=True)}


def features_at(days: dict, end: date) -> list[float] | None:
    """Weather for the 8 weeks up to `end` (inclusive). Same function shape as the live API's."""
    rows = [days.get((end - timedelta(i)).isoformat()) for i in range(56)]
    if any(r is None or r[0] is None or r[1] is None for r in rows):
        return None
    recent, before = np.array(rows[:28], dtype=float), np.array(rows[28:], dtype=float)
    week = end.isocalendar().week
    return [recent[:, 0].mean(), before[:, 0].mean(), recent[:, 1].mean(), before[:, 1].mean(), recent[:, 2].sum(), before[:, 2].sum(),
            float(np.sin(2 * np.pi * week / 52)), float(np.cos(2 * np.pi * week / 52))]


def load():
    rows = list(csv.DictReader(CASES.open(encoding="utf-8")))
    cache = {n: weather(n) for n in PLACES}
    x, y, year = [], [], []
    for r in rows:
        if r["exclude_training_row"] not in ("0", "False") or r["exclude_target_plus1"] not in ("0", "False") or r["target_outbreak_plus1"] == "":
            continue
        end = date.fromisocalendar(int(r["year"]), int(r["iso_week"]), 7)
        f = features_at(cache[r["district"]], end)
        if f is None:
            continue
        x.append(f)
        y.append(int(float(r["target_outbreak_plus1"])))
        year.append(int(r["year"]))
    return np.array(x), np.array(y), np.array(year)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def gb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42)


def year_folds(make, x, y, year):
    p = np.zeros(len(y))
    for yr in np.unique(year):
        m = make().fit(x[year != yr], y[year != yr])
        p[year == yr] = m.predict_proba(x[year == yr])[:, 1]
    return roc_auc_score(y, p)


def report(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {"n": len(y), "positives": int(y.sum()), "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3), "auc": round(float(roc_auc_score(y, p)), 3)}


def main() -> None:
    x, y, year = load()
    print(f"{len(y)} district-weeks, {int(y.sum())} above usual", {int(k): int((year == k).sum()) for k in np.unique(year)}, flush=True)
    tr, te = year <= 2021, year >= 2022
    a_lr, a_gb = year_folds(lr, x[tr], y[tr], year[tr]), year_folds(gb, x[tr], y[tr], year[tr])
    print(f"year-fold AUC: logistic {a_lr:.3f}, boosting {a_gb:.3f}", flush=True)
    make = gb if a_gb > a_lr + 0.02 else lr
    early = year <= 2020
    probe = make().fit(x[early], y[early])
    pv = probe.predict_proba(x[year == 2021])[:, 1]
    yv = y[year == 2021]
    t = float(max(v for v in np.unique(pv) if (pv[yv == 1] >= v).mean() >= TARGET_SENSITIVITY))
    model = make().fit(x[tr], y[tr])
    pt = model.predict_proba(x[te])[:, 1]
    metrics = {"test_2022_2023": report(y[te], pt, t), "threshold": round(t, 3), "model": make.__name__}
    print("test (once):", json.dumps(metrics), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = {"features": FEATURES, "threshold": t, "metrics": metrics, "train_min": x[tr].min(0).tolist(), "train_max": x[tr].max(0).tolist(),
            "trained_on": "Weekly dengue counts for 27 Karnataka districts, 2017-2021 (collected by BlackCatDisha/EpiSentinel), with Open-Meteo weather"}
    if make is lr:
        sc, clf = model.steps[0][1], model.steps[1][1]
        body |= {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    else:
        from skl2onnx import to_onnx
        onx = to_onnx(model, x[:1].astype(np.float32), options={id(model): {"zipmap": False}}, target_opset=17)
        OUT.with_suffix(".onnx").write_bytes(onx.SerializeToString())
        body["onnx_sha256"] = hashlib.sha256(OUT.with_suffix(".onnx").read_bytes()).hexdigest()
    OUT.write_text(json.dumps(body, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
