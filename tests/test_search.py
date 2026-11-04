"""Tests for Retriever using a fake store. No Qdrant, no model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from copilot.retrieval.search import METHODS, Retriever


def _hit(pid: str, score: float) -> MagicMock:
    h = MagicMock()
    h.payload = {"id": pid, "summary": f"summary-{pid}", "text": f"text-{pid}"}
    h.score = score
    return h


def _retriever(hits: list[MagicMock]) -> Retriever:
    embedders = MagicMock()
    embedders.dense.return_value = [[0.1] * 4]
    embedders.sparse.return_value = [{1: 0.5}]
    store = MagicMock()
    store.search_dense.return_value = hits
    store.search_sparse.return_value = hits
    store.search_hybrid.return_value = hits
    return Retriever(embedders=embedders, store=store)


@pytest.mark.parametrize("method", ["dense", "sparse", "hybrid"])
def test_methods_return_dicts(method: str):
    hits = [_hit("a", 0.9), _hit("b", 0.8)]
    r = _retriever(hits)
    out = r.search("q", collection="paragraphs_v1", method=method, limit=2)
    assert len(out) == 2
    assert out[0]["id"] == "a"
    assert "payload" in out[0]


def test_hybrid_rerank_uses_rerank_fn():
    hits = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
    r = _retriever(hits)
    r._rerank_fn = lambda q, ts: [0.1, 0.9, 0.5]  # b best, then c, then a
    out = r.search("q", collection="paragraphs_v1", method="hybrid_rerank", limit=2)
    assert [h["id"] for h in out] == ["b", "c"]


def test_unknown_method_raises():
    r = _retriever([])
    # Il codice sorgente solleva "Metodo sconosciuto: <method>".
    with pytest.raises(ValueError, match="sconosciuto"):
        r.search("q", collection="x", method="bogus")  # type: ignore[arg-type]


def test_methods_constant():
    # `dense_rerank` è stato rimosso intenzionalmente (vedi docstring di search.py).
    assert METHODS == ("dense", "sparse", "hybrid", "hybrid_rerank")


@pytest.mark.parametrize("method", METHODS)
def test_all_declared_methods_are_implemented(method: str):
    """Every method advertised in METHODS must be executable by Retriever.search."""
    hits = [_hit("a", 0.9), _hit("b", 0.8)]
    r = _retriever(hits)
    r._rerank_fn = lambda q, ts: [0.5] * len(ts)
    out = r.search("q", collection="paragraphs_v1", method=method, limit=2)
    assert len(out) >= 1