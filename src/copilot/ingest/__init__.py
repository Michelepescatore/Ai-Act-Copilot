"""Ingest scripts for the AI Act corpus.

Two independent submodules:

    chunk_paragraphs — deterministic, produces `data/processed/paragraphs.jsonl`
    chunk_articles   — uses the LLM, produces `data/processed/articles.jsonl`

Both are run as modules:

    python -m copilot.ingest.chunk_paragraphs
    python -m copilot.ingest.chunk_articles

We deliberately do not re-export their functions here: `python -m` imports
the parent package (`copilot.ingest`) before the target submodule, so an
eager `from copilot.ingest.chunk_articles import ...` in this file would
fail with ModuleNotFoundError. Callers import the submodule directly.
"""
