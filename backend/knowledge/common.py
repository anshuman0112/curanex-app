"""Shared helpers for building the Curanex library (free, openly licensed medical sources for the doctor module).

Every chunk keeps where it came from (source, title, section, url) and its licence, so every answer can cite it.
Raw downloads and the built library live in backend/data/knowledge/ (git-ignored).
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "knowledge"
RAW = DATA / "raw"
CHUNKS = DATA / "chunks"  # one .jsonl per source
MAX_WORDS, MIN_WORDS = 700, 40


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_words(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split a long section into pieces of at most max_words, cutting at sentence ends where possible."""
    sentences = re.split(r"(?<=[.!?])\s+", clean(text))
    out, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def chunk(source: str, title: str, section: str, url: str, licence: str, text: str, **extra) -> list[dict]:
    parts = [p for p in split_words(text) if len(p.split()) >= MIN_WORDS or extra.get("keep_short")]
    extra.pop("keep_short", None)
    return [{"source": source, "title": clean(title), "section": clean(section), "url": url, "licence": licence,
             "part": i + 1, "text": p, **extra} for i, p in enumerate(parts)]


def write_chunks(name: str, chunks: Iterable[dict]) -> int:
    CHUNKS.mkdir(parents=True, exist_ok=True)
    n = 0
    with (CHUNKS / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_chunks() -> list[dict]:
    out = []
    for path in sorted(CHUNKS.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out
