"""Search accuracy of the Curanex library on backend/tests/knowledge_cases.json (fixed before running, 2026-11-12).
A question counts as found when any of its top-5 StatPearls passages comes from a chapter whose title contains one of the expected words.
From backend/: python -m knowledge.eval_search
"""

import json
from pathlib import Path

from api.knowledge.search import search

CASES = Path(__file__).resolve().parents[1] / "tests" / "knowledge_cases.json"


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    hit1 = hit5 = 0
    misses = []
    for c in cases:
        titles = [r["title"].lower() for r in search(c["q"], k=5, sources=("StatPearls",))]
        ok = [any(e in t for e in c["expect"]) for t in titles]
        hit1 += bool(ok[:1] and ok[0])
        hit5 += any(ok)
        if not any(ok):
            misses.append((c["q"], titles[:3]))
    n = len(cases)
    print(f"questions {n} | recall@1 {hit1 / n:.3f} | recall@5 {hit5 / n:.3f}")
    for q, t in misses:
        print(" MISS:", q, "->", t)


if __name__ == "__main__":
    main()
