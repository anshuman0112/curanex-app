"""Build the Curanex library index from backend/data/knowledge/chunks/*.jsonl.
From backend/:
  python -m knowledge.ingest            # SQLite store + keyword index (FTS5) + embed_input.jsonl for the GPU step
  python -m knowledge.ingest --embed    # embed locally (fine for small libraries; the full library is embedded on Kaggle)

Outputs in backend/data/knowledge/:
  library.db       chunks (id, source, title, section, url, licence, part, text, extra) + chunks_fts (keyword search)
  embed_input.jsonl  {"id", "text"} in a fixed order, for the embedding step
  embeddings.npy + embedding_ids.json   (from Kaggle or --embed; float16, L2-normalised, all-MiniLM-L6-v2)
"""

import hashlib
import json
import sqlite3
import sys

import numpy as np

from knowledge.common import DATA, read_chunks

DB = DATA / "library.db"
EMBED_INPUT = DATA / "embed_input.jsonl"
EMB = DATA / "embeddings.npy"
EMB_IDS = DATA / "embedding_ids.json"
BASE = ("source", "title", "section", "url", "licence", "part", "text")


def chunk_id(c: dict) -> str:
    return hashlib.sha1(f"{c['source']}|{c['url']}|{c['section']}|{c['part']}|{c['text'][:80]}".encode()).hexdigest()[:16]


def embed_text(c: dict) -> str:
    return f"{c['title']}. {c['section']}. {c['text']}"


def build() -> list[dict]:
    chunks = read_chunks()
    seen, rows = set(), []
    for c in chunks:
        cid = chunk_id(c)
        if cid in seen:
            continue
        seen.add(cid)
        rows.append({**c, "id": cid})
    DATA.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source TEXT, title TEXT, section TEXT, url TEXT, licence TEXT, part INT, text TEXT, extra TEXT)")
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                    [(r["id"], *(r[k] for k in BASE), json.dumps({k: v for k, v in r.items() if k not in (*BASE, "id")}, ensure_ascii=False)) for r in rows])
    con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(title, section, text, content='chunks', content_rowid='rowid', tokenize='porter')")
    con.execute("INSERT INTO chunks_fts(rowid, title, section, text) SELECT rowid, title, section, text FROM chunks")
    con.commit()
    con.close()
    with EMBED_INPUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "text": embed_text(r)}, ensure_ascii=False) + "\n")
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"library.db: {len(rows)} chunks {by_source}; embed_input.jsonl written")
    return rows


def embed_local() -> None:
    from sentence_transformers import SentenceTransformer

    items = [json.loads(line) for line in EMBED_INPUT.open(encoding="utf-8")]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode([i["text"] for i in items], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMB, vecs.astype(np.float16))
    EMB_IDS.write_text(json.dumps([i["id"] for i in items]))
    print("embeddings", vecs.shape)


if __name__ == "__main__":
    build()
    if "--embed" in sys.argv:
        embed_local()
