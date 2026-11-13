"""Per-user storage in Supabase (owner, 2026-11-13: "every detail stored, chat history to reports, individually per user").

Server-side only, with the secret key from backend/.env (SUPABASE_URL, SUPABASE_SECRET_KEY). Everything lives in the private
Storage bucket BUCKET, one folder per user: <user_id>/<kind>/<time>-<id>.json for records (chats, results, reports, cases, plans)
and <user_id>/<file> for uploaded images and reports. (The API keys cannot create database tables, so records are JSON objects;
backend/supabase/schema.sql is the table version if a database password is added later.) Nothing here is public.
Saving never blocks or breaks the app: writes run in a background thread and failures are only logged.
"""
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from api.core.logging import log

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
BUCKET = "curanex-uploads"
TIMEOUT = 20


def _env(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    if "PYTEST_CURRENT_TEST" in os.environ or not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


def _conf() -> tuple[str, str]:
    return _env("SUPABASE_URL").rstrip("/"), _env("SUPABASE_SECRET_KEY")


def enabled() -> bool:
    url, key = _conf()
    return bool(url and key)


def _headers(extra: dict | None = None) -> dict:
    _, key = _conf()
    return {"apikey": key, "Authorization": f"Bearer {key}", **(extra or {})}


def _later(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True).start()


OWNER = {"doctor_plans": "patient_id"}  # whose folder a record belongs in (default user_id)


def _insert(table: str, row: dict) -> None:
    owner = row.get(OWNER.get(table, "user_id"))
    if owner is None:
        return
    now = time.time()
    record = {"id": uuid.uuid4().hex[:12], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), **row}
    name = f"{owner}/{table}/{int(now * 1000):014d}-{record['id']}.json"
    _upload(name, json.dumps(record, ensure_ascii=False, default=str).encode("utf-8"), "application/json")


def save(table: str, row: dict) -> None:
    """Save one record in the background (no-op when Supabase is not configured)."""
    if enabled():
        _later(_insert, table, row)


def _upload(path: str, data: bytes, mime: str) -> None:
    url, _ = _conf()
    try:
        r = httpx.post(f"{url}/storage/v1/object/{BUCKET}/{path}", content=data, headers=_headers({"Content-Type": mime, "x-upsert": "true"}), timeout=60)
        r.raise_for_status()
    except Exception as e:
        log.warning("cloud_upload_error", error=type(e).__name__, detail=getattr(getattr(e, "response", None), "text", "")[:200])


def save_file(user_id: int, name: str, data: bytes, mime: str = "application/octet-stream") -> str | None:
    """Store a file under <user_id>/<name> in the private bucket. Returns the stored path."""
    if not enabled():
        return None
    path = f"{user_id}/{name}"
    _later(_upload, path, data, mime)
    return path


def ensure_bucket() -> None:
    if not enabled():
        return
    url, _ = _conf()
    try:
        r = httpx.post(f"{url}/storage/v1/bucket", json={"id": BUCKET, "name": BUCKET, "public": False}, headers=_headers(), timeout=TIMEOUT)
        if r.status_code not in (200, 201, 400, 409):
            log.warning("cloud_bucket_error", status=r.status_code)
    except Exception as e:
        log.warning("cloud_bucket_error", error=type(e).__name__)


def rows(table: str, user_id: int, limit: int = 100, user_column: str = "user_id") -> list[dict]:
    """The user's own records of one kind, newest first (only files inside that user's folder are read)."""
    if not enabled():
        return []
    url, _ = _conf()
    try:
        r = httpx.post(f"{url}/storage/v1/object/list/{BUCKET}", json={"prefix": f"{int(user_id)}/{table}/", "limit": limit,
                       "sortBy": {"column": "name", "order": "desc"}}, headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        names = [o["name"] for o in r.json() if o.get("name", "").endswith(".json")]

        def read(name: str) -> dict | None:
            try:
                g = httpx.get(f"{url}/storage/v1/object/{BUCKET}/{int(user_id)}/{table}/{name}", headers=_headers(), timeout=TIMEOUT)
                return g.json() if g.status_code == 200 else None
            except Exception:
                return None

        with ThreadPoolExecutor(8) as pool:
            out = [x for x in pool.map(read, names) if x]
        return sorted(out, key=lambda x: x.get("created_at", ""), reverse=True)
    except Exception as e:
        log.warning("cloud_read_error", table=table, error=type(e).__name__)
        return []


def signed_url(path: str, seconds: int = 3600) -> str | None:
    """A temporary link so the owner can open their own file."""
    if not (enabled() and path):
        return None
    url, _ = _conf()
    try:
        r = httpx.post(f"{url}/storage/v1/object/sign/{BUCKET}/{path}", json={"expiresIn": seconds}, headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        return f"{url}/storage/v1{r.json()['signedURL']}"
    except Exception:
        return None
