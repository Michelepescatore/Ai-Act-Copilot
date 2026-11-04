"""Index `articles.jsonl` and `paragraphs.jsonl` into Qdrant.

Two collections:
    articles_v1    — semantic profile of each article/annex
    paragraphs_v1  — verbatim paragraph text

What we embed:
    articles_v1    — title + summary + key_conditions + paragraphs_summary
                     (semantic profile, good for "which article is about X")
    paragraphs_v1  — the verbatim text (good for exact citation retrieval)

Run:
    python -m copilot.retrieval.index
    python -m copilot.retrieval.index --only articles
    python -m copilot.retrieval.index --only paragraphs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from copilot.retrieval.embedders import Embedders
from copilot.retrieval.store import QdrantStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

ARTICLES_FILE = DATA_PROCESSED / "articles.jsonl"
PARAGRAPHS_FILE = DATA_PROCESSED / "paragraphs.jsonl"

ARTICLES_COLLECTION = "articles_v1"
PARAGRAPHS_COLLECTION = "paragraphs_v1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _article_text(record: dict) -> str:
    """Compose the text to embed for an article.

    We embed a synthetic "profile" (not the verbatim text) because the
    articles collection is queried to decide *which* article is about a
    given case. Summaries + conditions + paragraph summaries are a much
    better signal than raw legal prose.
    """
    parts = [record.get("title", "")]
    parts.append(record.get("summary", ""))
    if record.get("key_conditions"):
        parts.append("conditions: " + ", ".join(record["key_conditions"]))
    for num, s in sorted((record.get("paragraphs_summary") or {}).items()):
        parts.append(f"p{num}: {s}")
    return "\n".join(p for p in parts if p)


def _article_payload(record: dict) -> dict:
    return {
        "id": record["id"],
        "doc_type": record["doc_type"],
        "article": record.get("article"),
        "annex": record.get("annex"),
        "title": record.get("title", ""),
        "roles": record.get("roles", []),
        "tiers": record.get("tiers", []),
        "summary": record.get("summary", ""),
        "key_conditions": record.get("key_conditions", []),
        "paragraphs_summary": record.get("paragraphs_summary", {}),
        "verbatim_excerpt": record.get("verbatim_excerpt", ""),
        "url": record.get("url", ""),
    }


def _paragraph_payload(record: dict) -> dict:
    return {
        "id": record["id"],
        "doc_type": record["doc_type"],
        "article": record.get("article"),
        "annex": record.get("annex"),
        "paragraph": record.get("paragraph"),
        "chunk_id": record["chunk_id"],
        "text": record.get("text", ""),
        "article_title": record.get("article_title", ""),
        "url": record.get("url", ""),
        "tokens": record.get("tokens", 0),
    }


def _index(
    *,
    records: list[dict],
    collection: str,
    embedders: Embedders,
    store: QdrantStore,
    text_fn,
    payload_fn,
    batch_size: int = 32,
) -> int:
    texts = [text_fn(r) for r in records]
    total = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        dense = embedders.dense(batch)
        sparse = embedders.sparse(batch)
        store.upsert(
            collection,
            ids=[r["id"] for r in records[i : i + batch_size]],
            payloads=[payload_fn(r) for r in records[i : i + batch_size]],
            dense=dense,
            sparse=sparse,
        )
        total += len(batch)
        logger.info("%s: indexed %d/%d", collection, total, len(texts))
    return total


def index_articles(embedders: Embedders, store: QdrantStore) -> int:
    store.ensure_collection(ARTICLES_COLLECTION, embedders.dense_dim)
    records = _read_jsonl(ARTICLES_FILE)
    return _index(
        records=records,
        collection=ARTICLES_COLLECTION,
        embedders=embedders,
        store=store,
        text_fn=_article_text,
        payload_fn=_article_payload,
    )


def index_paragraphs(embedders: Embedders, store: QdrantStore) -> int:
    store.ensure_collection(PARAGRAPHS_COLLECTION, embedders.dense_dim)
    records = _read_jsonl(PARAGRAPHS_FILE)
    return _index(
        records=records,
        collection=PARAGRAPHS_COLLECTION,
        embedders=embedders,
        store=store,
        text_fn=lambda r: r["text"],
        payload_fn=_paragraph_payload,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["articles", "paragraphs"], default=None)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    embedders = Embedders()
    store = QdrantStore(url=args.qdrant_url)

    if args.only in (None, "articles"):
        n = index_articles(embedders, store)
        print(f"Indexed {n} articles -> {ARTICLES_COLLECTION}")
    if args.only in (None, "paragraphs"):
        n = index_paragraphs(embedders, store)
        print(f"Indexed {n} paragraphs -> {PARAGRAPHS_COLLECTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
