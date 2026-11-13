"""Watch data: /v1/wearables/apple-health (iPhone Health export upload) and /v1/wearables/sync (Android Health Connect summaries).
Both return the same daily summaries, fixed-rule alerts and 7-day averages; logged-in users' data is also saved to their private folder."""
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api import cloud, wearables
from api.auth.deps import optional_user

router = APIRouter(tags=["wearables"])
MAX_BYTES = 1024 * 1024 * 1024  # Health exports with years of data are big; the file is streamed, never loaded whole


class SyncIn(BaseModel):
    days: list[dict[str, Any]] = Field(default=[], max_length=60)
    source: str = Field(default="Health Connect", max_length=80)
    apps: list[str] = Field(default=[], max_length=20)


def _reply(user: dict | None, parsed: dict, via: str) -> dict:
    out = {**parsed, "via": via, "alerts": wearables.alerts(parsed["days"]), "summary": wearables.summary(parsed["days"])}
    if user and parsed["days"]:
        cloud.save("vitals", {"user_id": user["id"], "user_email": user["email"], "via": via, "sources": parsed.get("sources"), "days": parsed["days"]})
    return out


@router.post("/v1/wearables/apple-health")
async def apple_health(file: UploadFile = File(...), user: dict | None = Depends(optional_user)):
    if file.size and file.size > MAX_BYTES:
        raise HTTPException(status_code=415, detail="That file is larger than 1 GB.")
    try:
        parsed = await run_in_threadpool(wearables.parse_apple_health, file.file)
    except wearables.ExportRejected as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    if not parsed["days"]:
        raise HTTPException(status_code=415, detail="No steps, heart rate, sleep or oxygen data found in the last 30 days of this export. "
                                                    "Check that your watch app shares data with Apple Health.")
    return _reply(user, parsed, "Apple Health export")


@router.post("/v1/wearables/sync")
def sync(body: SyncIn, user: dict | None = Depends(optional_user)) -> dict:
    days = wearables.clean_days(body.days)
    return _reply(user, {"days": days, "sources": body.apps or [body.source], "latest": days[-1]["date"] if days else None}, body.source)


# ---- Live readings from an iPhone Shortcut (Nothing X -> Apple Health -> Shortcut every hour -> here) ----
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402

from fastapi import Header, Request  # noqa: E402

from api.auth import store  # noqa: E402
from api.auth.deps import current_user  # noqa: E402

LIVE: dict[int, dict] = {}  # user_id -> {"days": {...}, "live": {...}, "hr": [...], "spo2": [...]}
_live_lock = threading.Lock()
FIELDS = {"heart_rate": (25, 250), "spo2": (50, 100), "steps": (0, 100000), "resting_hr": (25, 200), "sleep_h": (0, 24),
          "hrv_ms": (0, 500), "active_kcal": (0, 10000), "respiratory_rate": (4, 60), "weight_kg": (10, 400)}


def _num(v) -> float | None:
    """Shortcuts may send 72, "72", "72 count/min", "0.97" (SpO2 as fraction) or "6 hr 30 min"."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    h = re.search(r"(\d+(?:\.\d+)?)\s*h", s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*min", s)
    if h or m:  # durations
        return round(float(h.group(1) if h else 0) + float(m.group(1) if m else 0) / 60, 2)
    f = re.search(r"-?\d+(?:[.,]\d+)?", s)
    return float(f.group(0).replace(",", ".")) if f else None


def _total(v) -> float | None:
    """Steps / calories arrive as many small samples (a list, or one number per line from Get Details > Value): add them up."""
    if isinstance(v, (list, tuple)):
        nums = [_num(x) for x in v]
        nums = [n for n in nums if n is not None]
        return float(sum(nums)) if nums else None
    if isinstance(v, str):
        t = re.sub(r"(?<=\d),(?=\d{3})", "", v)  # 1,234 -> 1234
        found = re.findall(r"-?\d+(?:\.\d+)?", t)
        return float(sum(float(x) for x in found)) if found else None
    return _num(v)


class DeviceKeyOut(BaseModel):
    key: str
    post_url_path: str


@router.post("/v1/wearables/device-key")
def device_key(user: dict = Depends(current_user)) -> DeviceKeyOut:
    """The key the iPhone Shortcut sends with each reading. Making a new one stops the old one."""
    return DeviceKeyOut(key=store.make_sync_key(user["id"]), post_url_path="/v1/wearables/live")


@router.post("/v1/wearables/live")
async def live(body: dict[str, Any], x_curanex_key: str | None = Header(default=None)) -> dict:
    user = store.user_for_sync_key(str(body.get("key") or x_curanex_key or "").strip())
    if not user:
        raise HTTPException(status_code=403, detail="Unknown Curanex sync key. Make a new one in Curanex > Connect watch.")
    got = {}
    for k, (lo, hi) in FIELDS.items():
        v = _total(body.get(k)) if k in ("steps", "active_kcal") else _num(body.get(k))
        if v is not None and k == "spo2" and v <= 1.0:
            v *= 100  # Apple Health stores 0.97
        if v is not None and lo <= v <= hi:
            got[k] = round(v, 1)
    if not got:
        raise HTTPException(status_code=422, detail="No readings in the request (heart_rate, spo2, steps, resting_hr, sleep_h ...).")
    today = date.today().isoformat()
    with _live_lock:
        st = LIVE.setdefault(user["id"], {"days": {}, "hr": {}, "spo2": {}, "live": {}})
        row = st["days"].setdefault(today, {"date": today})
        if "heart_rate" in got:
            hrs = st["hr"].setdefault(today, []); hrs.append(got["heart_rate"])
            row["hr_avg"], row["hr_max"] = round(sum(hrs) / len(hrs)), round(max(hrs))
        if "spo2" in got:
            sps = st["spo2"].setdefault(today, []); sps.append(got["spo2"])
            row["spo2_min"], row["spo2_avg"] = round(min(sps)), round(sum(sps) / len(sps))
        if "steps" in got:
            row["steps"] = max(row.get("steps", 0), got["steps"])  # today's total only grows
        for k, out in (("resting_hr", "hr_resting"), ("sleep_h", "sleep_h"), ("hrv_ms", "hrv_ms"), ("active_kcal", "kcal_active"), ("weight_kg", "weight_kg")):
            if k in got:
                row[out] = got[k]
        st["live"] = {**st["live"], **got, "at": time.time()}
        days = [st["days"][d] for d in sorted(st["days"])][-30:]
        live_now = dict(st["live"])
    cloud.save("vitals", {"user_id": user["id"], "user_email": user["email"], "via": "iPhone Shortcut (live)", "sources": ["Nothing X via Apple Health"],
                          "days": days, "live": live_now})
    return {"ok": True, "saved": sorted(got), "live": live_now, "alerts": wearables.alerts(days)}


@router.get("/v1/wearables/state")
def state(user: dict = Depends(current_user)) -> dict:
    """Latest live readings + daily rows for the logged-in user (the app polls this while open)."""
    with _live_lock:
        st = LIVE.get(user["id"])
        if not st:
            return {"days": [], "live": None, "alerts": []}
        days = [st["days"][d] for d in sorted(st["days"])][-30:]
        return {"days": days, "live": dict(st["live"]), "alerts": wearables.alerts(days), "via": "iPhone Shortcut (live)",
                "sources": ["Nothing X via Apple Health"]}


@router.get("/v1/wearables/live")
async def live_get(request: Request) -> dict:
    """Opening the link in a browser shows that it works. A Shortcut left on GET also works: readings may come as ?key=...&heart_rate=..."""
    q = dict(request.query_params)
    if not q.get("key"):
        return {"ok": True, "message": "This is your Curanex watch link and it is working. It is meant for your iPhone Shortcut, which sends your readings here. "
                                        "Nothing to do in the browser: set the Shortcut's method to POST (or add ?key=YOUR_KEY&heart_rate=72&spo2=97&steps=3000)."}
    return await live(q, None)
