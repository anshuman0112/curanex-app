"""Area alerts: the live feature summary matches training, offline weather is handled, and the estimate stays experimental."""

import json
from datetime import date, timedelta
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from api import alerts
from main import app
from training.train_dengue_weather import features_at as training_features

client = TestClient(app)


def fake_days(end: date, temp=28.0, humid=80.0, rain=5.0):
    return {(end - timedelta(i)).isoformat(): (temp + i % 3, humid, rain) for i in range(70)}


def test_live_features_match_training():
    end = date(2024, 8, 4)
    days = fake_days(end)
    assert alerts.features_at(days, end) == training_features(days, end)


def test_missing_district_is_explained():
    r = client.get("/alerts", params={"district": ""})
    assert r.status_code == 422 and "district" in r.json()["detail"]


def _handler(temp, humid, rain):
    def handle(request: httpx.Request) -> httpx.Response:
        if "geocoding" in request.url.host:
            return httpx.Response(200, json={"results": [{"name": "Mysuru", "admin1": "Karnataka", "latitude": 12.3, "longitude": 76.6}]})
        end = date(2024, 8, 4)
        times = [(end - timedelta(i)).isoformat() for i in range(alerts.DAYS, -2, -1)]
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": times,
                    "temperature_2m_mean": [temp] * len(times),
                    "relative_humidity_2m_mean": [humid] * len(times),
                    "precipitation_sum": [rain] * len(times),
                }
            },
        )

    return handle


def run(temp, humid, rain, district="Mysuru"):
    alerts._cache.clear()
    real = httpx.Client
    with patch("api.alerts.httpx.Client", lambda: real(transport=httpx.MockTransport(_handler(temp, humid, rain)))):
        return client.get("/alerts", params={"district": district})


def test_monsoon_weather_gives_advice_and_an_experimental_estimate():
    body = run(27.0, 85.0, 6.0).json()
    assert body["mosquito_weather"] is True and len(body["advice"]) == 4
    assert body["weather"]["rain_4_weeks_mm"] == 168
    assert body["estimate"] is None or body["estimate"]["experimental"] is True
    assert "not a diagnosis" in body["notice"]


def test_dry_cold_weather_is_not_called_mosquito_weather():
    body = run(12.0, 30.0, 0.0, "Leh").json()
    assert body["mosquito_weather"] is False
    assert body["estimate"] is None  # far outside the Karnataka training weather


def test_weather_service_down_gives_a_message():
    alerts._cache.clear()
    real = httpx.Client
    down = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("down")))
    with patch("api.alerts.httpx.Client", lambda: real(transport=down)):
        r = client.get("/alerts", params={"district": "Pune"})
    assert r.status_code == 422 and "weather service" in r.json()["detail"]


def test_model_file_is_pinned():
    import hashlib

    meta = json.loads(alerts.META.read_text(encoding="utf-8"))
    assert hashlib.sha256(alerts.META.with_suffix(".onnx").read_bytes()).hexdigest() == meta["onnx_sha256"]
