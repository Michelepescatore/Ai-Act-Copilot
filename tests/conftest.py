"""Shared fixtures for the LangGraph graph tests.

They provide:
- `FakeLLM`: LLM that responds with predefined outputs depending on the
  schema and the content of the prompt. Records every call.
- `FakeRetriever`: retriever that returns predefined hits, without Qdrant.
- `patch_retriever`: monkeypatch of `nodes._retriever` to use `FakeRetriever`.

These fakes allow testing the graph end-to-end without a real LLM or
Qdrant, in a few milliseconds.
"""

from __future__ import annotations

from typing import Any

import pytest

from copilot.llm import LLMError


class FakeLLM:
    """Fake LLM that responds based on the requested Pydantic schema.

    You pass a list of `responses` (dicts to validate against the schema)
    that will be returned in order, one per call to `structured_complete`.
    If a response is an exception, it is raised.

    Records every call in `self.calls`, so tests can verify what was
    passed to the LLM (prompt, system, schema).
    """

    def __init__(self, responses: list[dict[str, Any] | Exception]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def structured_complete(self, user_prompt, schema, *, system_prompt=""):
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "schema": schema.__name__,
            }
        )
        if not self._responses:
            raise AssertionError(
                f"FakeLLM: no response prepared for {schema.__name__}"
            )
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response)


class FakeRetriever:
    """Fake retriever: returns predefined hits based on the collection."""

    def __init__(self, hits_by_collection: dict[str, list[dict]]):
        self._hits = hits_by_collection
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, collection, method="hybrid", limit=10):
        self.calls.append(
            {"query": query, "collection": collection, "method": method, "limit": limit}
        )
        return self._hits.get(collection, [])[:limit]


@pytest.fixture
def fake_llm():
    """Returns a factory: `fake_llm([resp1, resp2, ...])`."""
    return FakeLLM


@pytest.fixture
def fake_retriever():
    """Returns a factory: `fake_retriever({"paragraphs_v1": [...], ...})`."""
    return FakeRetriever


@pytest.fixture
def patch_retriever(monkeypatch):
    """Replace `nodes._retriever` with a FakeRetriever.

    Usage:
        patch_retriever(FakeRetriever({"paragraphs_v1": [...]}))

    Imports `copilot.graph.nodes` lazily: if the module is missing, only
    the tests that use this fixture will fail, not the whole `conftest.py`.

    Note: `_retriever` is decorated with `functools.lru_cache`, so it has a
    `.cache_clear()` method. We use `monkeypatch.setattr` and let pytest
    restore the original at teardown — we do NOT call `.cache_clear()` after
    the monkeypatch, because at that point `_retriever` is a lambda and no
    longer has `cache_clear`.
    """
    import copilot.graph.nodes as graph_nodes

    def _patch(retriever):
        # Clear the cache of the real singleton BEFORE replacing it.
        graph_nodes._retriever.cache_clear()
        # Replace the function; monkeypatch will restore the original.
        monkeypatch.setattr(graph_nodes, "_retriever", lambda: retriever)
        return retriever

    yield _patch

    # Restore the original and clear the cache (the original has cache_clear).
    monkeypatch.undo()
    graph_nodes._retriever.cache_clear()