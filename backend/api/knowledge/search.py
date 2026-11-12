"""Curanex library search for the doctor module: keyword (SQLite FTS5, BM25) + meaning (all-MiniLM-L6-v2 vectors), merged by
reciprocal-rank fusion. Every hit carries its source, title, section, url and licence so answers can cite it.
Built by backend/knowledge/ingest.py into backend/data/knowledge/. Works with keyword search alone if vectors are missing.
"""

import json
import re
import sqlite3
import threading
from functools import lru_cache

import numpy as np

from knowledge.common import DATA

DB = DATA / "library.db"
EMB = DATA / "embeddings.npy"
EMB_IDS = DATA / "embedding_ids.json"
RRF_K = 60
_lock = threading.Lock()
STOP = {"the", "and", "for", "with", "from", "this", "that", "have", "has", "had", "are", "was", "were", "what", "which", "patient", "days"}


@lru_cache(maxsize=1)
def _vectors() -> tuple[np.ndarray, list[str]] | None:
    if not (EMB.exists() and EMB_IDS.exists()):
        return None
    return np.load(EMB).astype(np.float32), json.loads(EMB_IDS.read_text())


@lru_cache(maxsize=1)
def _embedder():
    from api.orchestrator.retriever import get_embedder

    return get_embedder()


def _connect() -> sqlite3.Connection:
    if not DB.exists():
        raise FileNotFoundError("The Curanex library is not built yet: run `python -m knowledge.ingest` in backend/.")
    con = sqlite3.connect(DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _fts_query(text: str) -> str:
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower()) if w not in STOP]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words)) or '""'


def _row(r: sqlite3.Row) -> dict:
    out = {k: r[k] for k in ("id", "source", "title", "section", "url", "licence", "part", "text")}
    out.update(json.loads(r["extra"] or "{}"))
    return out


def search(query: str, k: int = 8, sources: tuple[str, ...] | None = None, pool: int = 60) -> list[dict]:
    """Top-k passages for the query. Each result has `score` (fused rank) and `similarity` (cosine, None without vectors)."""
    con = _connect()
    with _lock:
        where = f" AND c.source IN ({','.join('?' * len(sources))})" if sources else ""
        kw = con.execute(
            f"SELECT c.id FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid WHERE chunks_fts MATCH ?{where} ORDER BY bm25(chunks_fts) LIMIT ?",
            (_fts_query(query), *(sources or ()), pool)).fetchall()
    ranks: dict[str, float] = {}
    for i, r in enumerate(kw):
        ranks[r["id"]] = ranks.get(r["id"], 0) + 1 / (RRF_K + i + 1)
    sims: dict[str, float] = {}
    vec = _vectors()
    embed = _embedder() if vec else None
    if vec and embed:
        matrix, ids = vec
        q = np.asarray(embed([query]), dtype=np.float32)[0]
        scores = matrix @ q
        allowed = None
        if sources:
            with _lock:
                allowed = {r["id"] for r in con.execute(f"SELECT id FROM chunks WHERE source IN ({','.join('?' * len(sources))})", sources)}
        order = [i for i in np.argsort(-scores) if allowed is None or ids[i] in allowed][:pool]
        for rank, i in enumerate(order):
            ranks[ids[i]] = ranks.get(ids[i], 0) + 1 / (RRF_K + rank + 1)
            sims[ids[i]] = float(scores[i])
    best = sorted(ranks, key=ranks.get, reverse=True)[:k]
    if not best:
        return []
    with _lock:
        rows = {r["id"]: _row(r) for r in con.execute(f"SELECT * FROM chunks WHERE id IN ({','.join('?' * len(best))})", best)}
    con.close()
    return [{**rows[i], "score": round(ranks[i], 5), "similarity": sims.get(i)} for i in best if i in rows]


def drug_sections(drug: str, fields: tuple[str, ...]) -> list[dict]:
    """Exact label sections for one medicine (copied text only). Empty if the medicine has no label in the library."""
    con = _connect()
    with _lock:
        rows = [_row(r) for r in con.execute("SELECT * FROM chunks WHERE source LIKE 'Drug label%' AND lower(json_extract(extra, '$.drug')) = ?",
                                              (drug.strip().lower(),))]
    con.close()
    return sorted([r for r in rows if r.get("field") in fields], key=lambda r: (fields.index(r["field"]), r["part"]))
