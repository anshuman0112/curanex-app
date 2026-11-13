"""Per-user storage in Supabase (owner, 2026-11-13: "every detail stored, chat history to reports, individually per user").

Server-side only, with the secret key from backend/.env (SUPABASE_URL, SUPABASE_SECRET_KEY). Tables: backend/supabase/schema.sql
(row level security on, no public access). Files go to the private bucket BUCKET under <user_id>/.
Saving never blocks or breaks the app: writes run in a background thread and failures are only logged.
"""
import os
import threading
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


def _insert(table: str, row: dict) -> None:
    url, _ = _conf()
    try:
        r = httpx.post(f"{url}/rest/v1/{table}", json=row, headers=_headers({"Prefer": "return=minimal"}), timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        log.warning("cloud_insert_error", table=table, error=type(e).__name__, detail=getattr(getattr(e, "response", None), "text", "")[:200])


def save(table: str, row: dict) -> None:
    """Insert one row in the background (no-op when Supabase is not configured)."""
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
    """The user's own rows, newest first."""
    if not enabled():
        return []
    url, _ = _conf()
    try:
        r = httpx.get(f"{url}/rest/v1/{table}", params={user_column: f"eq.{user_id}", "order": "created_at.desc", "limit": str(limit)},
                      headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
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
