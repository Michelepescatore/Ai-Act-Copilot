# FILE: store.py
"""Thin wrapper around Qdrant.

Two collections:
    articles_v1    — one point per article/annex (semantic profile)
    paragraphs_v1  — one point per paragraph (verbatim text)

Each collection stores both a dense and a sparse vector, so we can
query with dense, sparse, hybrid, or hybrid + rerank.

The wrapper exposes only what the indexer and the retriever need:
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
        # Test seam: inject a fake client; no network call.
        self.client = client or QdrantClient(url=url)

    def ensure_collection(self, name: str, dense_dim: int) -> None:
        """Create the collection with both vectors, dense and sparse.

        Idempotent: if the collection already exists, do nothing.
        Also creates payload indices for efficient filtering.
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
        # Create payload indices for the fields we want to filter on.
        # This speeds up filtered queries (e.g. step 5 of the pipeline).
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
        """Upsert the points. All lists must have the same length.

        Qdrant expects the sparse vector as `SparseVector(indices, values)`,
        not as a dict `{int: float}`. We convert here, so callers can keep
        the Python-friendly format.
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
        """Hybrid = dense + sparse fused with Reciprocal Rank Fusion (RRF).

        We increase the number of prefetch candidates to give the fusion
        more material to work with (your point 4: with limit=5 the
        prefetch was taking only 10 candidates, too few).
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
    """Qdrant point IDs must be int or UUID. We hash strings to int64.

    Using the same string always produces the same int, so upserts are
    idempotent. Collisions are astronomically unlikely (sha256 → 63 bits).
    """
    digest = hashlib.sha256(point_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)