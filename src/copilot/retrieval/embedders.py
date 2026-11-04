"""Dense + sparse embeddings with bge-m3.

bge-m3 produces both a dense vector (for semantic search) and a sparse
vector (for lexical/keyword search) from the same model, so we load it once
and use it for both.

Lazy loading: the model (~2 GB) is downloaded the first time an `Embedders()`
instance is created **without injected functions**. Subsequent instances
reuse the module-level singleton (see `_MODEL_CACHE`), so the model is
loaded only once per process.

Testing: pass `dense_fn=` and `sparse_fn=` to override the model entirely.
In that case the singleton is not touched and no model is loaded.
"""

from __future__ import annotations

from collections.abc import Callable

_DENSE_DIM = 1024  # bge-m3 dense output dimension

# Module-level cache: model_name → BGEM3FlagModel instance.
# Created lazily on first use, reused by every Embedders() that does not
# inject custom functions.
_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_name: str):
    """Return the cached BGEM3FlagModel for `model_name`, loading it if needed."""
    if model_name not in _MODEL_CACHE:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore

        _MODEL_CACHE[model_name] = BGEM3FlagModel(model_name, use_fp16=False)
    return _MODEL_CACHE[model_name]


class Embedders:
    """Dense and sparse embeddings from a single bge-m3 model."""

    def __init__(
        self,
        *,
        dense_fn: Callable[[list[str]], list[list[float]]] | None = None,
        sparse_fn: Callable[[list[str]], list[dict[int, float]]] | None = None,
        model_name: str = "BAAI/bge-m3",
    ) -> None:
        # Test seam: if both functions are injected, we never touch the model.
        if dense_fn is not None and sparse_fn is not None:
            self._dense_fn = dense_fn
            self._sparse_fn = sparse_fn
            return

        # Lazy loading via module-level cache: the model is loaded once per
        # (process, model_name), regardless of how many Embedders() instances
        # are created.
        self.model = _get_model(model_name)
        self._dense_fn = dense_fn or self._default_dense
        self._sparse_fn = sparse_fn or self._default_sparse

    def _default_dense(self, texts: list[str]) -> list[list[float]]:
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"].tolist()

    def _default_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        out = self.model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # bge-m3 returns {token_id_str: weight}. Qdrant wants {int: float}.
        return [{int(k): float(v) for k, v in lex.items()} for lex in out["lexical_weights"]]

    @property
    def dense_dim(self) -> int:
        return _DENSE_DIM

    def dense(self, texts: list[str]) -> list[list[float]]:
        return self._dense_fn(texts)

    def sparse(self, texts: list[str]) -> dict[int, float]:
        return self._sparse_fn(texts)