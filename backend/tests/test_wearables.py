"""Watch data: Apple Health export parsing (iPhone + Nothing X), Android sync, fixed alert rules."""
import io
import zipfile
from datetime import date

from api import wearables

XML = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_IN">
 <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Nothing X" unit="count" startDate="2026-11-12 08:00:00 +0530" endDate="2026-11-12 09:00:00 +0530" value="3000"/>
 <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Nothing X" unit="count" startDate="2026-11-12 18:00:00 +0530" endDate="2026-11-12 19:00:00 +0530" value="2500"/>
 <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone 17 Pro" unit="count" startDate="2026-11-12 08:00:00 +0530" endDate="2026-11-12 09:00:00 +0530" value="4000"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Nothing X" unit="count/min" startDate="2026-11-12 10:00:00 +0530" endDate="2026-11-12 10:00:00 +0530" value="70"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Nothing X" unit="count/min" startDate="2026-11-12 11:00:00 +0530" endDate="2026-11-12 11:00:00 +0530" value="110"/>
 <Record type="HKQuantityTypeIdentifierRestingHeartRate" sourceName="Nothing X" unit="count/min" startDate="2026-11-12 06:00:00 +0530" endDate="2026-11-12 06:00:00 +0530" value="64"/>
 <Record type="HKQuantityTypeIdentifierOxygenSaturation" sourceName="Nothing X" unit="%" startDate="2026-11-12 03:00:00 +0530" endDate="2026-11-12 03:00:00 +0530" value="0.89"/>
 <Record type="HKQuantityTypeIdentifierOxygenSaturation" sourceName="Nothing X" unit="%" startDate="2026-11-12 04:00:00 +0530" endDate="2026-11-12 04:00:00 +0530" value="0.97"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Nothing X" startDate="2026-11-11 23:30:00 +0530" endDate="2026-11-12 03:30:00 +0530" value="HKCategoryValueSleepAnalysisAsleepCore"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Nothing X" startDate="2026-11-12 03:30:00 +0530" endDate="2026-11-12 05:00:00 +0530" value="HKCategoryValueSleepAnalysisAsleepDeep"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Nothing X" startDate="2026-11-11 23:00:00 +0530" endDate="2026-11-12 06:00:00 +0530" value="HKCategoryValueSleepAnalysisInBed"/>
 <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Nothing X" unit="count" startDate="2026-09-01 08:00:00 +0530" endDate="2026-09-01 09:00:00 +0530" value="999"/>
 <Workout workoutActivityType="HKWorkoutActivityTypeWalking"/>
</HealthData>"""


def _zip() -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("apple_health_export/export.xml", XML)
    return b.getvalue()


def test_export_zip_is_summarised_per_day():
    out = wearables.parse_apple_health(io.BytesIO(_zip()), today=date(2026, 11, 13))
    day = next(d for d in out["days"] if d["date"] == "2026-11-12")
    assert day["steps"] == 5500  # watch 5,500 vs phone 4,000: the bigger source, never added together
    assert day["hr_avg"] == 90 and day["hr_max"] == 110 and day["hr_resting"] == 64
    assert day["spo2_min"] == 89 and day["spo2_avg"] == 93
    assert day["sleep_h"] == 5.5  # asleep core + deep; "in bed" not counted
    assert "Nothing X" in out["sources"] and all(d["date"] >= "2026-10-14" for d in out["days"])  # only the last 30 days


def test_low_oxygen_alert_is_a_fixed_rule():
    out = wearables.parse_apple_health(io.BytesIO(_zip()), today=date(2026, 11, 13))
    a = wearables.alerts(out["days"])
    assert a[0]["level"] == "urgent" and "89%" in a[0]["text"]


def test_bad_files_are_refused():
    import pytest
    with pytest.raises(wearables.ExportRejected):
        wearables.parse_apple_health(io.BytesIO(b"not a health export"))
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("other.txt", "x")
    with pytest.raises(wearables.ExportRejected):
        wearables.parse_apple_health(io.BytesIO(b.getvalue()))


def test_upload_and_android_sync_endpoints(client):
    r = client.post("/v1/wearables/apple-health", files={"file": ("export.zip", _zip(), "application/zip")})
    assert r.status_code == 200 and r.json()["alerts"][0]["level"] == "urgent"
    s = client.post("/v1/wearables/sync", json={"days": [{"date": "2026-11-12", "steps": 6240, "hr_resting": 82, "sleep_h": 5.2, "spo2_min": 96, "junk": 1},
                                                         {"date": "bad", "steps": 1}], "source": "Health Connect", "apps": ["Nothing X"]}).json()
    assert s["days"] == [{"date": "2026-11-12", "steps": 6240, "hr_resting": 82, "sleep_h": 5.2, "spo2_min": 96}]
    assert s["summary"]["latest_day"]["steps"] == 6240


def test_live_readings_from_shortcut(client):
    from api.routers import wearables as w
    from tests.test_auth_records import signup
    w.LIVE.clear()
    headers, _ = signup(client, "live@watch.test", "patient")
    key = client.post("/v1/wearables/device-key", headers=headers).json()["key"]
    assert key.startswith("cx_")
    assert client.post("/v1/wearables/live", json={"key": "cx_wrong", "heart_rate": 70}).status_code == 403
    r = client.post("/v1/wearables/live", json={"key": key, "heart_rate": "72 count/min", "spo2": "0.97", "steps": 4200, "sleep_h": "6 hr 30 min"}).json()
    assert r["live"]["heart_rate"] == 72 and r["live"]["spo2"] == 97 and r["live"]["sleep_h"] == 6.5
    client.post("/v1/wearables/live", json={"key": key, "heart_rate": 110, "spo2": 0.89, "steps": 3000}, )
    s = client.get("/v1/wearables/state", headers=headers).json()
    day = s["days"][-1]
    assert day["hr_avg"] == 91 and day["hr_max"] == 110 and day["spo2_min"] == 89 and day["steps"] == 4200  # steps never go down
    assert s["alerts"][0]["level"] == "urgent"
    new_key = client.post("/v1/wearables/device-key", headers=headers).json()["key"]
    assert client.post("/v1/wearables/live", json={"key": key, "heart_rate": 70}).status_code == 403  # old key stops working
    assert client.post("/v1/wearables/live", headers={"X-Curanex-Key": new_key}, json={"heart_rate": 70}).status_code == 200


def test_live_link_is_friendly_in_a_browser_and_works_with_get(client):
    from api.routers import wearables as w
    from tests.test_auth_records import signup
    w.LIVE.clear()
    r = client.get("/v1/wearables/live")
    assert r.status_code == 200 and "working" in r.json()["message"]
    headers, _ = signup(client, "get@watch.test", "patient")
    key = client.post("/v1/wearables/device-key", headers=headers).json()["key"]
    g = client.get(f"/v1/wearables/live?key={key}&heart_rate=72&spo2=97&steps=3000").json()
    assert g["live"]["heart_rate"] == 72 and g["live"]["spo2"] == 97
    assert client.get("/v1/wearables/live?key=cx_wrong&heart_rate=70").status_code == 403
