"""Build `data/processed/paragraphs.jsonl` from `data/processed/ai_act.json`.

Pure deterministic: no LLM. One record per paragraph of an article, one per
block of an annex.

Long paragraphs are NOT split. Each paragraph stays as one atomic chunk, even
if it exceeds `MAX_TOKENS`. This keeps the corpus aligned with the legal
document: a paragraph is a paragraph, as written. It also removes the
artificial `.s1`, `.s2`, ... suffixes that polluted the index and complicated
the evaluation (the ground truth in `cases.jsonl` already points to parent
paragraph IDs, so no normalization is needed anymore).

Trade-off: some paragraphs are very long (e.g. `art_5.p1`, `art_26.p10`),
so their dense/sparse embedding is "spread" over more text and the reranker
has to find the relevant sub-part in a larger context. This is acceptable
for the current pipeline, and can be revisited by reintroducing recursive
splitting if retrieval quality on those specific paragraphs drops.

The chunk preserves the article/paragraph identifiers:
    chunk_id = "art_10.p2"   (always, no suffix)

Why two doc types:
  - Articles carry substantive obligations (Art. 9, Art. 10, ...).
  - Annex blocks carry substantive content too (Annex III defines high-risk
    use cases). Both need to be indexed in the same collection.

Output schema (per record):

    article:
        {
          "id": "art_10.p2",
          "doc_type": "article",
          "article": "10",
          "annex": null,
          "paragraph": "2",
          "chunk_id": "art_10.p2",
          "text": "...",
          "article_title": "Data and data governance",
          "url": "...",
          "tokens": 187
        }

    annex block:
        {
          "id": "anx_III.b4",
          "doc_type": "annex",
          "article": null,
          "annex": "III",
          "paragraph": null,
          "chunk_id": "anx_III.b4",
          "text": "...",
          "article_title": "ANNEX III - High-risk AI systems...",
          "url": "...",
          "tokens": 120
        }

Placeholder paragraphs like "Article 4" are skipped: they carry no content
and would pollute the index (any "Article X" query would match them).

Run:
    python -m copilot.ingest.chunk_paragraphs
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Paths and constants


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # -> backend/
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "ai_act.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "paragraphs.jsonl"

CORPUS_CELEX = "02024R1689-20260727"
BASE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"

# Reference threshold, no longer used to split. Kept as documentation:
# paragraphs above this size are the ones that used to be split into
# `.s1`, `.s2`, ... and that are now kept as single chunks.
MAX_TOKENS = 500

# Below this length, an unnumbered paragraph is a placeholder like "Article 4".
# Real intros (e.g. Art. 3 "For the purposes of this Regulation…") are longer.
_MIN_PLACEHOLDER_CHARS = 40


# Token counting


_TOKENIZER: Any = None


def _count_tokens(text: str) -> int:
    # Count tokens with the bge-m3 tokenizer (loaded lazily, cached by HF).
    # Tests patch this function to avoid the ~2 GB download.
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


# Record builders


def _is_placeholder(paragraph: dict) -> bool:
    # True for unnumbered paragraphs whose text is just the article label.
    if paragraph.get("number") is not None:
        return False
    text = (paragraph.get("text") or "").strip()
    return len(text) < _MIN_PLACEHOLDER_CHARS


def _article_url(base_url: str, article_number: str) -> str:
    return f"{base_url}#art_{article_number}"


def _annex_url(base_url: str, annex_number: str) -> str:
    return f"{base_url}#anx_{annex_number}"


def _paragraph_records(
    *,
    paragraph: dict,
    article: dict,
    base_url: str,
    count_tokens: Callable[[str], int],
) -> list[dict]:
    # Exactly one record per non-placeholder, non-empty paragraph.
    # No splitting: `id` and `chunk_id` are always the original paragraph id.
    if _is_placeholder(paragraph):
        return []

    text = (paragraph.get("text") or "").strip()
    if not text:
        return []

    return [
        {
            "id": paragraph["id"],
            "doc_type": "article",
            "article": article["number"],
            "annex": None,
            "paragraph": paragraph.get("number"),
            "chunk_id": paragraph["id"],
            "text": text,
            "article_title": article.get("title", ""),
            "url": _article_url(base_url, article["number"]),
            "tokens": count_tokens(text),
        }
    ]


def _annex_block_record(
    *,
    block: dict,
    annex: dict,
    base_url: str,
    count_tokens: Callable[[str], int],
) -> dict | None:
    # One record per annex block. Returns None if the block is empty.
    text = (block.get("text") or "").strip()
    if not text:
        return None

    block_id = f"{annex['id']}.b{block['n']}"
    return {
        "id": block_id,
        "doc_type": "annex",
        "article": None,
        "annex": annex["number"],
        "paragraph": None,
        "chunk_id": block_id,
        "text": text,
        "article_title": annex.get("title", ""),
        "url": _annex_url(base_url, annex["number"]),
        "tokens": count_tokens(text),
    }


# Main walk


def generate_paragraph_chunks(
    doc: dict,
    *,
    count_tokens: Callable[[str], int] | None = None,
) -> list[dict]:
    # Walk the parsed document and return all paragraph + annex-block records.
    count_tokens = count_tokens or _count_tokens
    base_url = BASE_URL.format(celex=CORPUS_CELEX)
    chunks: list[dict] = []

    # Articles live either directly under a chapter or under a section.
    for chapter in doc.get("chapters", []):
        article_groups = [chapter.get("articles", [])]
        article_groups += [s.get("articles", []) for s in chapter.get("sections", [])]
        for articles in article_groups:
            for article in articles:
                for paragraph in article.get("paragraphs", []):
                    chunks.extend(
                        _paragraph_records(
                            paragraph=paragraph,
                            article=article,
                            base_url=base_url,
                            count_tokens=count_tokens,
                        )
                    )

    for annex in doc.get("annexes", []):
        for block in annex.get("blocks", []):
            rec = _annex_block_record(
                block=block,
                annex=annex,
                base_url=base_url,
                count_tokens=count_tokens,
            )
            if rec is not None:
                chunks.append(rec)

    return chunks


# CLI


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}. Run parse_html.py first.")

    doc = json.loads(args.input.read_text(encoding="utf-8"))
    chunks = generate_paragraph_chunks(doc)
    _write_jsonl(chunks, args.output)

    n_art = sum(1 for c in chunks if c["doc_type"] == "article")
    n_anx = sum(1 for c in chunks if c["doc_type"] == "annex")
    # Con il nuovo schema non ci sono più chunk `.sN`: la metrica è 0 e serve
    # solo come sanity check che la rimozione dello split sia effettiva.
    n_split = sum(1 for c in chunks if ".s" in c["chunk_id"])
    print(
        f"Paragraphs: {len(chunks)} "
        f"(article: {n_art}, annex: {n_anx}, split: {n_split})  ->  {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())