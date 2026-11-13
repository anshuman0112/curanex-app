"""Smartwatch data from the watch's own app (owner, 2026-11-13: no Curanex band; read what the user's watch app already collected).

iPhone (free path): watch app (e.g. Nothing X) -> Apple Health -> Health app "Export All Health Data" -> export.zip uploaded here.
Android: watch app -> Health Connect -> Curanex Android app sends daily summaries to /v1/wearables/sync.
Both end as the same daily summary. Alert rules are fixed code (never the AI). Values are shown as context, not a diagnosis.
"""
import io
import re
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from xml.etree.ElementTree import iterparse

DAYS = 30
FIELDS = ("steps", "hr_resting", "hr_avg", "hr_max", "hrv_ms", "sleep_h", "spo2_min", "spo2_avg", "kcal_active", "weight_kg", "bp_sys", "bp_dia")

# Apple Health record types we keep -> our field
TYPES = {
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierHeartRate": "hr",
    "HKQuantityTypeIdentifierRestingHeartRate": "hr_resting",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_ms",
    "HKQuantityTypeIdentifierOxygenSaturation": "spo2",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "kcal_active",
    "HKQuantityTypeIdentifierBodyMass": "weight_kg",
    "HKQuantityTypeIdentifierBloodPressureSystolic": "bp_sys",
    "HKQuantityTypeIdentifierBloodPressureDiastolic": "bp_dia",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
}
ASLEEP = re.compile(r"Asleep", re.I)  # AsleepCore / AsleepDeep / AsleepREM / AsleepUnspecified / Asleep (not InBed / Awake)


class ExportRejected(ValueError):
    pass


def _when(s: str) -> datetime:
    # "2026-11-12 23:10:00 +0530" -> local wall time (the day the user lived it)
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def _xml_stream(fileobj):
    """export.xml from the Health app's export.zip (or a bare export.xml)."""
    head = fileobj.read(4)
    fileobj.seek(0)
    if head[:2] == b"PK":
        z = zipfile.ZipFile(fileobj)
        names = [n for n in z.namelist() if n.lower().endswith("/export.xml") or n.lower() == "export.xml"]
        if not names:
            raise ExportRejected("This zip has no export.xml. In the Health app use Profile > Export All Health Data, then upload export.zip.")
        return z.open(names[0])
    if head[:1] == b"<" or head[:3] == b"\xef\xbb\xbf":
        return fileobj
    raise ExportRejected("Please upload the export.zip file from the Health app.")


def parse_apple_health(fileobj, today: date | None = None) -> dict:
    """Stream-parse an Apple Health export (works on very large files). Returns {"days": [...], "sources": [...], "latest": "YYYY-MM-DD"}."""
    stream = _xml_stream(fileobj)
    per_source: dict = defaultdict(lambda: defaultdict(float))  # (day, field, source) sums for steps / kcal
    samples: dict = defaultdict(list)                            # (day, field) -> values
    sleep: dict = defaultdict(float)                             # (day, source) -> hours
    sources: set = set()
    latest = None
    try:
        for _, el in iterparse(stream, events=("end",)):
            if el.tag != "Record":
                if el.tag in ("Correlation", "Workout", "ActivitySummary"):
                    el.clear()
                continue
            field = TYPES.get(el.get("type", ""))
            if field:
                try:
                    start, end = _when(el.get("startDate", "")), _when(el.get("endDate", ""))
                except ValueError:
                    el.clear()
                    continue
                src = el.get("sourceName") or "Unknown"
                if field == "sleep":
                    if ASLEEP.search(el.get("value", "")):
                        sleep[(end.date(), src)] += max(0.0, (end - start).total_seconds() / 3600)
                        sources.add(src)
                        latest = max(latest or end.date(), end.date())
                else:
                    try:
                        v = float(el.get("value", ""))
                    except ValueError:
                        el.clear()
                        continue
                    d = start.date()
                    if field == "spo2" and v <= 1.0:
                        v *= 100  # Apple stores 0.97
                    if field == "weight_kg" and (el.get("unit") or "").lower() == "lb":
                        v *= 0.4536
                    if field in ("steps", "kcal_active"):
                        per_source[(d, field)][src] += v
                    else:
                        samples[(d, field)].append(v)
                    sources.add(src)
                    latest = max(latest or d, d)
            el.clear()
    except Exception as e:  # broken / truncated file
        if isinstance(e, ExportRejected):
            raise
        raise ExportRejected("The export file could not be read. Please export again from the Health app.") from e

    if latest is None:
        return {"days": [], "sources": [], "latest": None}
    end_day = min(latest, today or latest)
    first = end_day - timedelta(days=DAYS - 1)
    days = []
    for i in range(DAYS):
        d = first + timedelta(days=i)
        row: dict = {"date": d.isoformat()}
        for field in ("steps", "kcal_active"):  # phone and watch both count steps: take the source with the most, never add them up
            by_src = per_source.get((d, field))
            if by_src:
                row[field] = round(max(by_src.values()))
        hr = samples.get((d, "hr"))
        if hr:
            row["hr_avg"], row["hr_max"] = round(sum(hr) / len(hr)), round(max(hr))
        for field, out in (("hr_resting", "hr_resting"), ("hrv_ms", "hrv_ms"), ("weight_kg", "weight_kg"), ("bp_sys", "bp_sys"), ("bp_dia", "bp_dia")):
            vals = samples.get((d, field))
            if vals:
                row[out] = round(sum(vals) / len(vals), 1)
        spo2 = samples.get((d, "spo2"))
        if spo2:
            row["spo2_min"], row["spo2_avg"] = round(min(spo2)), round(sum(spo2) / len(spo2))
        hours = [h for (day, _), h in sleep.items() if day == d]
        if hours:
            row["sleep_h"] = round(min(max(hours), 16), 1)  # one source's night (watch + phone would double it)
        if len(row) > 1:
            days.append(row)
    return {"days": days, "sources": sorted(sources), "latest": end_day.isoformat()}


def clean_days(days: list) -> list[dict]:
    """Daily summaries sent by the Android app: keep known fields, sane ranges, last 30 days."""
    limits = {"steps": 100000, "hr_resting": 250, "hr_avg": 250, "hr_max": 250, "hrv_ms": 500, "sleep_h": 24, "spo2_min": 100, "spo2_avg": 100,
              "kcal_active": 10000, "weight_kg": 400, "bp_sys": 300, "bp_dia": 200}
    out = []
    for d in (days or [])[-DAYS:]:
        if not isinstance(d, dict) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(d.get("date", ""))):
            continue
        row = {"date": d["date"]}
        for k, cap in limits.items():
            try:
                v = float(d[k])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= v <= cap:
                row[k] = v
        out.append(row)
    return out


def alerts(days: list[dict]) -> list[dict]:
    """Fixed safety rules on watch data (no AI). Newest problems first."""
    out = []
    recent = [d for d in days if d][-7:]
    for d in reversed(recent):
        if d.get("spo2_min") is not None and d["spo2_min"] < 92:
            out.append({"level": "urgent", "date": d["date"], "text": f"Your watch recorded low oxygen ({round(d['spo2_min'])}%) on {d['date']}. "
                        "If you feel breathless, have chest pain or feel confused, use SOS or call 112. Otherwise see a doctor soon. Watch readings can be wrong, so recheck at rest."})
            break
    for d in reversed(recent):
        r = d.get("hr_resting")
        if r is not None and (r > 100 or r < 40):
            out.append({"level": "warn", "date": d["date"], "text": f"Your resting heart rate was {round(r)} bpm on {d['date']}, outside the usual 50-100. Please see a doctor, sooner if you feel dizzy, faint or breathless."})
            break
    prev = [d["hr_resting"] for d in days[-14:-7] if d.get("hr_resting") is not None]
    last = [d["hr_resting"] for d in days[-7:] if d.get("hr_resting") is not None]
    if len(prev) >= 3 and len(last) >= 3 and sum(last) / len(last) - sum(prev) / len(prev) >= 10:
        out.append({"level": "warn", "date": days[-1]["date"], "text": "Your resting heart rate this week is 10+ bpm higher than last week. This can come with illness, poor sleep or stress. If it continues, see a doctor."})
    short = [d["sleep_h"] for d in days[-7:] if d.get("sleep_h") is not None]
    if len(short) >= 4 and sum(short) / len(short) < 6:
        out.append({"level": "info", "date": days[-1]["date"], "text": f"You slept about {sum(short) / len(short):.1f} hours a night this week. Adults need 7-9 hours."})
    return out


def summary(days: list[dict]) -> dict:
    """Today + 7-day averages, for the AI doctor and the home screen."""
    if not days:
        return {}
    last7 = days[-7:]

    def avg(k):
        v = [d[k] for d in last7 if d.get(k) is not None]
        return round(sum(v) / len(v), 1) if v else None

    return {"latest_day": days[-1], "avg_7d": {k: avg(k) for k in FIELDS if avg(k) is not None}}


def parse_bytes(data: bytes) -> dict:
    return parse_apple_health(io.BytesIO(data))
