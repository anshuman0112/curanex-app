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
