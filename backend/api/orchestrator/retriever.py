"""Model search (WORKFLOW.md Part C): embed each routable model's `description` with all-MiniLM-L6-v2,
embed the user's message (+ last 2 turns), cosine similarity, keep top-k at or above the threshold.
Description embeddings are cached on disk and rebuilt when any description changes."""
import hashlib
import threading
from collections.abc import Callable

import numpy as np

from api.core.config import get_settings
from api.core.logging import log
from api.registry import routable_models

Embedder = Callable[[list[str]], np.ndarray]
_embedder: Embedder | None = None
_memo: dict[str, tuple[list[str], np.ndarray]] = {}
_load_lock = threading.Lock()


def _default_embedder() -> Embedder | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning("model_search_disabled", reason="sentence-transformers not installed")
        return None
    model = SentenceTransformer(get_settings().embedding_model)
    return lambda texts: model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def set_embedder(fn: Embedder | None) -> None:
    """Swap the embedder (tests). None resets to the default."""
    global _embedder
    _embedder = fn
    _memo.clear()


def get_embedder() -> Embedder | None:
    global _embedder
    with _load_lock:  # warm-up and the first request must not load the model twice
        if _embedder is None:
            _embedder = _default_embedder()
    return _embedder


def _model_matrix(models: list[dict], embed: Embedder) -> np.ndarray:
    ids = [m["id"] for m in models]
    key = hashlib.sha256("\n".join(f"{m['id']}\t{m.get('description', '')}" for m in models).encode()).hexdigest()
    if key in _memo:
        return _memo[key][1]
    cache = get_settings().embeddings_cache
    matrix = None
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if str(data["key"]) == key:
            matrix = data["matrix"]
    if matrix is None:
        matrix = np.asarray(embed([m.get("description", "") for m in models]), dtype=np.float32)
        try:
            np.savez(cache, key=np.array(key), matrix=matrix)
        except OSError:
            pass
    _memo.clear()
    _memo[key] = (ids, matrix)
    return matrix


def warm_up() -> None:
    """Load the embedding model and the description matrix ahead of the first question."""
    try:
        models = routable_models()
        embed = get_embedder() if models else None
        if embed is not None:
            _model_matrix(models, embed)
            log.info("model_search_ready", models=len(models))
    except Exception as e:  # never block startup
        log.warning("model_search_warmup_failed", error=type(e).__name__)


def search(text: str, top_k: int | None = None, threshold: float | None = None,
           context: str = "") -> list[tuple[dict, float]]:
    """Score the message alone and the message with recent turns; each model keeps its better score,
    so a long earlier reply can't drown out what the user is asking now."""
    s = get_settings()
    models = routable_models()
    embed = get_embedder() if models and text.strip() else None
    if embed is None:
        return []
    matrix = _model_matrix(models, embed)
    queries = [text] + ([f"{context} {text}"] if context.strip() else [])
    q = np.asarray(embed(queries), dtype=np.float32)
    scores = (matrix @ q.T).max(axis=1)
    cutoff = s.retrieval_threshold if threshold is None else threshold
    said = f"{context} {text}".lower()
    # A model can list words that rule it out (e.g. adult diabetes checks are not for pregnancy); those never fire.
    blocked = {i for i, m in enumerate(models) if any(w in said for w in m.get("exclude_if") or [])}
    order = [i for i in np.argsort(-scores) if i not in blocked][: top_k or s.retrieval_top_k]
    return [(models[i], float(scores[i])) for i in order if scores[i] >= cutoff]
