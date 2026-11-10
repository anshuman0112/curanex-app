"""Health alert for the user's area: dengue-season weather for a district, from Open-Meteo (free, no key).

Only the district name the user typed is sent here; no GPS position is asked for or stored, and the district is not logged.
Results are cached per district for 12 hours, which does the job of the nightly refresh in ALL_MODELS_PLAN Route D.

Two parts, kept apart in the response:
- weather: measured facts (temperature, humidity, rain over the last 4 weeks) and whether they suit Aedes mosquitoes,
  using commonly cited ranges. This is general guidance, not a prediction.
- estimate: a weather-only model trained on Karnataka district counts. It scored AUC 0.67 on 2022-2023, below the
  Curanex bar, so it is marked experimental and omitted when the weather is outside what it was trained on.
"""

import json
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort

META = Path(__file__).resolve().parent / "alert_models" / "dengue_weather.json"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
DAILY = "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
TTL_S = 12 * 3600
DAYS = 56
# Aedes mosquitoes breed and bite most in warm, humid weather after rain.
WARM = (20.0, 34.0)
HUMID = 60.0
WET_MM = 20.0
ADVICE = [
    "Empty standing water every week: coolers, flower pots, tyres, buckets.",
    "Use mosquito repellent and wear long sleeves, especially early morning and evening.",
    "Use nets or window screens. Aedes mosquitoes bite during the day.",
    "See a doctor if you get a high fever with body ache, pain behind the eyes or a rash. "
    "Avoid ibuprofen and aspirin until dengue is ruled out.",
]
NOTICE = "Area information, not a diagnosis. Based on weather, not on reported cases."
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()
_session: dict = {}


class AlertError(ValueError):
    """Shown to the user as is."""


def features_at(days: dict, end: date) -> list[float] | None:
    """The same 8-week weather summary the model was trained on (training/train_dengue_weather.py)."""
    rows = [days.get((end - timedelta(i)).isoformat()) for i in range(DAYS)]
    if any(r is None or r[0] is None or r[1] is None for r in rows):
        return None
    recent, before = np.array(rows[:28], dtype=float), np.array(rows[28:], dtype=float)
    week = end.isocalendar().week
    return [
        recent[:, 0].mean(),
        before[:, 0].mean(),
        recent[:, 1].mean(),
        before[:, 1].mean(),
        recent[:, 2].sum(),
        before[:, 2].sum(),
        float(np.sin(2 * np.pi * week / 52)),
        float(np.cos(2 * np.pi * week / 52)),
    ]


def _place(district: str, client: httpx.Client) -> dict:
    names = [district, district.replace(" district", "").replace(" Urban", "").replace(" Rural", ""), district.split(",")[0]]
    for name in dict.fromkeys(n.strip() for n in names if n.strip()):
        r = client.get(GEOCODE, params={"name": name, "count": 5, "countryCode": "IN", "language": "en"}, timeout=15)
        hits = r.json().get("results") or [] if r.status_code == 200 else []
        if hits:
            h = hits[0]
            return {"name": h["name"], "state": h.get("admin1", ""), "lat": round(h["latitude"], 2), "lon": round(h["longitude"], 2)}
    raise AlertError("I couldn't find that district. Check the spelling in your profile, for example 'Mysuru' or 'Pune'.")


def _estimate(meta: dict, f: list[float]) -> dict | None:
    lo, hi = np.array(meta["train_min"]), np.array(meta["train_max"])
    x = np.array(f)
    span = hi - lo
    if np.any(x[:6] < lo[:6] - 0.1 * span[:6]) or np.any(x[:6] > hi[:6] + 0.1 * span[:6]):
        return None  # weather unlike anything in training: no estimate
    with _lock:
        if "s" not in _session:
            _session["s"] = ort.InferenceSession(str(META.with_suffix(".onnx")), providers=["CPUExecutionProvider"])
    s = _session["s"]
    p = float(s.run(None, {s.get_inputs()[0].name: x[None].astype(np.float32)})[1][0][1])
    test = meta["metrics"]["test_2022_2023"]
    return {
        "level": "Higher than usual" if p >= meta["threshold"] else "Usual for this time of year",
        "experimental": True,
        "accuracy_note": f"Experimental: on 2022-2023 Karnataka data it caught {round(100 * test['sensitivity'])}% of above-usual weeks "
        f"with many false alarms (AUC {test['auc']}).",
        "trained_on": meta["trained_on"],
    }


def alert(district: str) -> dict:
    district = " ".join(district.split())[:60]
    if len(district) < 2:
        raise AlertError("Add your district in your profile to see alerts for your area.")
    key = district.lower()
    with _lock:
        hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL_S:
        return hit[1]
    try:
        with httpx.Client() as client:
            place = _place(district, client)
            r = client.get(
                FORECAST,
                params={
                    "latitude": place["lat"],
                    "longitude": place["lon"],
                    "past_days": DAYS,
                    "forecast_days": 1,
                    "daily": DAILY,
                    "timezone": "Asia/Kolkata",
                },
                timeout=20,
            )
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise AlertError("The weather service isn't reachable right now. Please try again later.") from e
    d = r.json()["daily"]
    days = {
        t: (a, b, c or 0.0)
        for t, a, b, c in zip(d["time"], d["temperature_2m_mean"], d["relative_humidity_2m_mean"], d["precipitation_sum"], strict=True)
    }
    end = date.fromisoformat(d["time"][-2]) if len(d["time"]) > 1 else date.today()  # yesterday: today is not complete yet
    f = features_at(days, end)
    if f is None:
        raise AlertError("Weather for your area is incomplete right now. Please try again later.")
    temp, humid, rain = round(float(f[0]), 1), round(float(f[2])), round(float(f[4]))
    suits = WARM[0] <= temp <= WARM[1] and humid >= HUMID and rain >= WET_MM
    meta = json.loads(META.read_text(encoding="utf-8"))
    out = {
        "district": district,
        "place": f"{place['name']}, {place['state']}".strip(", "),
        "weather": {"temperature_c": temp, "humidity_pct": humid, "rain_4_weeks_mm": rain, "until": end.isoformat()},
        "mosquito_weather": suits,
        "headline": "Weather now suits dengue mosquitoes" if suits else "Weather is less suited to dengue mosquitoes right now",
        "estimate": _estimate(meta, f) if META.with_suffix(".onnx").exists() else None,
        "advice": ADVICE if suits else ADVICE[:1] + ADVICE[3:],
        "notice": NOTICE,
        "source": "Weather: Open-Meteo (open-meteo.com)",
    }
    with _lock:
        _cache[key] = (time.time(), out)
    return out
