"""Answer cache for the doctor module (SQLite, backend/data/knowledge/cache.db).
Calls are deterministic (temperature 0, same passages), so an identical request returns the stored reply instantly instead of
calling the AI again. Prompts also keep a fixed system prefix first, so a provider that supports prompt caching can reuse it.
"""

import hashlib
import json
import sqlite3
import threading
import time

from knowledge.common import DATA

DB = DATA / "cache.db"
TTL_S = 30 * 24 * 3600
_lock = threading.Lock()
stats = {"hits": 0, "misses": 0}


def _con() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, reply TEXT, created REAL)")
    return con


def key_for(model: str, system: str, user: str) -> str:
    return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode()).hexdigest()


def get(key: str) -> dict | None:
    with _lock:
        con = _con()
        row = con.execute("SELECT reply, created FROM llm_cache WHERE key = ?", (key,)).fetchone()
        con.close()
    if row and time.time() - row[1] < TTL_S:
        stats["hits"] += 1
        return json.loads(row[0])
    stats["misses"] += 1
    return None


def put(key: str, reply: dict) -> None:
    with _lock:
        con = _con()
        con.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)", (key, json.dumps(reply, ensure_ascii=False), time.time()))
        con.commit()
        con.close()
