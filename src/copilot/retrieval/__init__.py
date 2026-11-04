"""Retrieval layer: embedders, Qdrant store, and hybrid retriever.

Re-exports the public API so callers can do:
    from copilot.retrieval import Embedders, QdrantStore, Retriever
"""

from copilot.retrieval.embedders import Embedders
from copilot.retrieval.search import Retriever
from copilot.retrieval.store import QdrantStore

__all__ = ["Embedders", "QdrantStore", "Retriever"]