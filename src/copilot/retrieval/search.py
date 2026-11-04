# FILE: search.py
"""Four retrieval methods based on QdrantStore.

    dense         — semantic similarity on the dense vector
    sparse        — lexical similarity on the sparse vector
    hybrid        — RRF fusion of dense and sparse (on the Qdrant side)   ← default
    hybrid_rerank — hybrid, then reranked on the client side with bge-reranker

The reranker is loaded lazily, like the embedders. Tests can inject a fake
reranker via `rerank_fn=`.

Note: `dense_rerank` was removed because it produces nearly identical results
to `hybrid_rerank` (same reranker, same function, same top-K), but it doubles
the evaluation time on CPU. If a comparison between "rerank on a dense basis"
and "rerank on a hybrid basis" is needed in the future, it will be enough to
reintroduce the block and add "dense_rerank" to METHODS.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from copilot.retrieval.embedders import Embedders
from copilot.retrieval.store import QdrantStore

Method = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
METHODS: tuple[Method, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")

# How many candidates to retrieve from the store before reranking.
# Higher = the reranker has more material to promote, but slower.
RERANK_CANDIDATE_MULTIPLIER = 3

# Default reranker model. Change here to switch from base to v2-m3.
#   "BAAI/bge-reranker-base"    → ~278M param, mono-lingual, faster
#   "BAAI/bge-reranker-v2-m3"   → ~568M param, multilingual, more accurate
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Retriever:
    def __init__(
        self,
        *,
        embedders: Embedders,
        store: QdrantStore,
        rerank_fn: Callable[[str, list[str]], list[float]] | None = None,
        reranker_model: str = RERANKER_MODEL,
    ) -> None:
        self.embedders = embedders
        self.store = store
        self._reranker_model = reranker_model

        if rerank_fn is not None:
            self._rerank_fn = rerank_fn
        else:
            self._rerank_fn = None  # loaded lazily

    def _rerank(self, query: str, texts: list[str]) -> list[float]:
        """Perform the reranking, loading the model on first use."""
        if self._rerank_fn is None:
            from FlagEmbedding import FlagReranker

            model = FlagReranker(self._reranker_model, use_fp16=False)

            # `compute_score` may return a float if there is only one text.
            # We always normalize it to a list of floats.
            def _score(q: str, ts: list[str]) -> list[float]:
                if not ts:
                    return []
                scores = model.compute_score([[q, t] for t in ts])
                if isinstance(scores, (int, float)):
                    return [float(scores)]
                return [float(s) for s in scores]

            self._rerank_fn = _score
        return self._rerank_fn(query, texts)

    def _get_text_for_rerank(self, hit) -> str:
        """
        Extract the text to pass to the reranker from the payload of a hit.
        Uses 'text' for paragraphs and 'title' + 'summary' for articles.
        """
        payload = hit.payload
        if text := payload.get("text"):
            return text
        title = payload.get("title", "")
        summary = payload.get("summary", "")
        if title or summary:
            return f"{title}\n{summary}".strip()
        return ""

    def search(
        self,
        query: str,
        *,
        collection: str,
        method: Method = "hybrid",
        limit: int = 10,
    ) -> list[dict]:
        """Return a list of dictionaries: {id, score, payload}."""

        def _as_dict(h, score: float) -> dict:
            """Convert a Qdrant Hit object into our standard format.
            Uses the original ID from the payload (string), not the internal hash ID."""
            return {
                "id": h.payload.get("id"),
                "score": float(score),
                "payload": h.payload,
            }

        if method == "dense":
            dense = self.embedders.dense([query])[0]
            hits = self.store.search_dense(collection, dense, limit=limit)
            return [_as_dict(h, h.score) for h in hits]

        if method == "sparse":
            sparse = self.embedders.sparse([query])[0]
            hits = self.store.search_sparse(collection, sparse, limit=limit)
            return [_as_dict(h, h.score) for h in hits]

        if method == "hybrid":
            dense = self.embedders.dense([query])[0]
            sparse = self.embedders.sparse([query])[0]
            hits = self.store.search_hybrid(collection, dense=dense, sparse=sparse, limit=limit)
            return [_as_dict(h, h.score) for h in hits]

        if method == "hybrid_rerank":
            dense = self.embedders.dense([query])[0]
            sparse = self.embedders.sparse([query])[0]
            candidates = limit * RERANK_CANDIDATE_MULTIPLIER
            hits = self.store.search_hybrid(
                collection, dense=dense, sparse=sparse, limit=candidates
            )
            if not hits:
                return []
            texts = [self._get_text_for_rerank(h) for h in hits]
            scores = self._rerank(query, texts)
            reranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
            return [_as_dict(h, s) for h, s in reranked[:limit]]

        raise ValueError(f"Unknown method: {method}")