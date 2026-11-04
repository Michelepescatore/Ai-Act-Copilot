# FILE: store.py
"""Wrapper sottile attorno a Qdrant.

Due collezioni:
    articles_v1    — un punto per ogni articolo/allegato (profilo semantico)
    paragraphs_v1  — un punto per ogni paragrafo (testo verbatim)

Ogni collezione memorizza sia un vettore denso che uno sparso, così possiamo
interrogare con dense, sparse, hybrid, o hybrid + rerank.

Il wrapper espone solo ciò di cui l'indicizzatore e il retriever hanno bisogno:
    ensure_collection, upsert, search_dense, search_sparse, search_hybrid.
"""

from __future__ import annotations

import hashlib
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantStore:
    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        client: Any | None = None,
    ) -> None:
        # Seam per i test: inietta un client finto; nessuna chiamata di rete.
        self.client = client or QdrantClient(url=url)

    def ensure_collection(self, name: str, dense_dim: int) -> None:
        """Crea la collezione con entrambi i vettori, denso e sparso.

        Idempotente: se la collezione esiste già, non fa nulla.
        Crea anche gli indici di payload per un filtraggio efficiente.
        """
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        # Crea indici di payload per i campi su cui vogliamo filtrare.
        # Questo velocizza le query con filtro (es. per step 5 della pipeline).
        for field_name in ["article", "doc_type", "annex"]:
            self.client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def upsert(
        self,
        collection: str,
        *,
        ids: list[str],
        payloads: list[dict],
        dense: list[list[float]],
        sparse: list[dict[int, float]],
    ) -> None:
        """Upsert dei punti. Tutte le liste devono avere la stessa lunghezza.

        Qdrant si aspetta il vettore sparso come `SparseVector(indices, values)`,
        non come un dict `{int: float}`. Convertiamo qui, così i chiamanti
        mantengono il formato Python-friendly.
        """
        assert len(ids) == len(payloads) == len(dense) == len(sparse)
        points = [
            PointStruct(
                id=_stable_id(point_id),
                vector={
                    DENSE_VECTOR_NAME: dense[i],
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=list(sparse[i].keys()),
                        values=list(sparse[i].values()),
                    ),
                },
                payload=payloads[i],
            )
            for i, point_id in enumerate(ids)
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search_dense(
        self,
        collection: str,
        vector: list[float],
        *,
        limit: int = 10,
        query_filter: Any | None = None,
    ) -> list[Any]:
        return self.client.query_points(
            collection_name=collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points

    def search_sparse(
        self,
        collection: str,
        vector: dict[int, float],
        *,
        limit: int = 10,
        query_filter: Any | None = None,
    ) -> list[Any]:
        return self.client.query_points(
            collection_name=collection,
            query=SparseVector(indices=list(vector.keys()), values=list(vector.values())),
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points

    def search_hybrid(
        self,
        collection: str,
        *,
        dense: list[float],
        sparse: dict[int, float],
        limit: int = 10,
        query_filter: Any | None = None,
    ) -> list[Any]:
        """Hybrid = dense + sparse fusi con Reciprocal Rank Fusion (RRF).

        Aumentiamo il numero di candidati di prefetch per dare alla fusione
        più materiale su cui lavorare (il tuo punto 4: con limit=5 il
        prefetch prendeva solo 10 candidati, troppo pochi).
        """
        return self.client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense, using=DENSE_VECTOR_NAME, limit=limit * 4),
                Prefetch(
                    query=SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
                    using=SPARSE_VECTOR_NAME,
                    limit=limit * 4,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points


def _stable_id(point_id: str) -> int:
    """Gli ID dei punti di Qdrant devono essere int o UUID. Hashiamo le stringhe a int64.

    Usare la stessa stringa produce sempre lo stesso int, quindi gli upsert sono
    idempotenti. Le collisioni sono astronomicamente improbabili (sha256 → 63 bit).
    """
    digest = hashlib.sha256(point_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)