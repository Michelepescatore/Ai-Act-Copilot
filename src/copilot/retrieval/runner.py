"""CLI runner: run a query against a collection with every method.

Prints a compact table and saves the full results as JSON to `data/evals/`.

Usage:
    python -m copilot.retrieval.runner \\
        --query "system for CV screening and candidate selection" \\
        --collection paragraphs_v1 \\
        --methods dense,sparse,hybrid,hybrid_rerank \\
        --limit 5 \\
        --out data/evals/run_2026-10-30_01.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from copilot.retrieval.embedders import Embedders
from copilot.retrieval.search import METHODS, Method, Retriever
from copilot.retrieval.store import QdrantStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVALS_DIR = PROJECT_ROOT / "data" / "evals"


def _print_hits(method: str, hits: list[dict], limit: int) -> None:
    print(f"\n=== {method} (top {min(limit, len(hits))}) ===")
    for rank, hit in enumerate(hits[:limit], start=1):
        p = hit["payload"]
        label = p.get("chunk_id") or p.get("id") or "?"
        title = p.get("article_title") or p.get("title") or ""
        score = hit["score"]
        print(f"{rank:>2}. {label:<14} score={score:.4f}  {title[:70]}")


def run(
    *,
    query: str,
    collection: str,
    methods: list[Method],
    limit: int,
    qdrant_url: str,
) -> dict:
    embedders = Embedders()
    store = QdrantStore(url=qdrant_url)
    retriever = Retriever(embedders=embedders, store=store)

    results: dict[str, list[dict]] = {}
    for method in methods:
        hits = retriever.search(query, collection=collection, method=method, limit=limit)
        results[method] = hits
        _print_hits(method, hits, limit)

    return {
        "query": query,
        "collection": collection,
        "limit": limit,
        "methods": methods,
        "timestamp": datetime.now(UTC).isoformat(),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--collection",
        default="paragraphs_v1",
        choices=["articles_v1", "paragraphs_v1"],
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help=f"Comma-separated subset of {METHODS}",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the JSON result. Default: data/evals/run_<ts>.json",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    methods: list[Method] = []
    for m in args.methods.split(","):
        m = m.strip()
        if m not in METHODS:
            parser.error(f"Unknown method: {m}. Choose from {METHODS}.")
        methods.append(m)  # type: ignore[arg-type]

    payload = run(
        query=args.query,
        collection=args.collection,
        methods=methods,
        limit=args.limit,
        qdrant_url=args.qdrant_url,
    )

    out = args.out
    if out is None:
        DEFAULT_EVALS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = DEFAULT_EVALS_DIR / f"run_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
