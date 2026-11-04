# FILE: search.py
"""Quattro metodi di retrieval basati su QdrantStore.

    dense         — similarità semantica sul vettore denso
    sparse        — similarità lessicale sul vettore sparso
    hybrid        — fusione RRF di dense e sparse (lato Qdrant)   ← default
    hybrid_rerank — hybrid, poi reranked lato client con bge-reranker

Il reranker viene caricato in modo lazy, come gli embedder. I test possono
iniettare un finto reranker tramite `rerank_fn=`.

Nota: `dense_rerank` è stato rimosso perché produce risultati quasi identici
a `hybrid_rerank` (stesso reranker, stessa funzione, stessa top-K), ma
raddoppia il tempo di valutazione su CPU. Se in futuro servirà un confronto
tra "rerank su base dense" e "rerank su base hybrid", basterà reintrodurre
il blocco e aggiungere "dense_rerank" a METHODS.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from copilot.retrieval.embedders import Embedders
from copilot.retrieval.store import QdrantStore

Method = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
METHODS: tuple[Method, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")

# Quanti candidati recuperare dallo store prima del reranking.
# Più alto = il reranker ha più materiale da cui promuovere, ma più lento.
RERANK_CANDIDATE_MULTIPLIER = 3

# Modello di reranker di default. Cambia qui per passare da base a v2-m3.
#   "BAAI/bge-reranker-base"    → ~278M param, mono-lingua, più veloce
#   "BAAI/bge-reranker-v2-m3"   → ~568M param, multilingue, più accurato
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
            self._rerank_fn = None  # caricato in modo lazy

    def _rerank(self, query: str, texts: list[str]) -> list[float]:
        """Esegue il reranking, caricando il modello al primo utilizzo."""
        if self._rerank_fn is None:
            from FlagEmbedding import FlagReranker

            model = FlagReranker(self._reranker_model, use_fp16=False)

            # `compute_score` può restituire un float se c'è un solo testo.
            # Lo standardizziamo sempre a una lista di float.
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
        Estrae il testo da passare al reranker dal payload di un hit.
        Usa 'text' per i paragrafi e 'title' + 'summary' per gli articoli.
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
        """Restituisce una lista di dizionari: {id, score, payload}."""

        def _as_dict(h, score: float) -> dict:
            """Converte un oggetto Hit di Qdrant nel nostro formato standard.
            Usa l'ID originale del payload (stringa), non l'ID hash interno."""
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

        raise ValueError(f"Metodo sconosciuto: {method}")