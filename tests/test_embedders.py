"""Tests for Embedders using injected functions. No model download."""

from __future__ import annotations

from copilot.retrieval.embedders import Embedders


def test_dense_and_sparse_use_injected_functions():
    emb = Embedders(
        dense_fn=lambda texts: [[0.1] * 1024 for _ in texts],
        sparse_fn=lambda texts: [{1: 0.5} for _ in texts],
    )
    assert len(emb.dense(["a", "b"])) == 2
    assert len(emb.sparse(["a", "b"])) == 2
    assert emb.dense_dim == 1024
