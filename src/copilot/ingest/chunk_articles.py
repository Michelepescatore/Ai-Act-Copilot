"""Build `data/processed/articles.jsonl` from `data/processed/ai_act.json`.

For each article and each annex, produce one record with semantic metadata:

    - `summary`, `key_conditions`, `paragraphs_summary`  -> LLM
    - `roles`, `tiers`                                    -> LLM, then validated
    - `verbatim_excerpt`, `url`, `id`                     -> deterministic

The LLM is told to pick `roles` and `tiers` only from closed vocabularies.
We re-filter after the call: any value outside the allow-list is dropped.
That way a hallucinated role ("chief AI officer") never reaches the index.

Output schema (per record):

    article:
        {
          "id": "art_10",
          "doc_type": "article",
          "article": "10",
          "annex": null,
          "title": "Data and data governance",
          "roles": ["provider"],
          "tiers": ["high_risk"],
          "summary": "...",
          "key_conditions": ["uses_training_data", "processes_personal_data"],
          "paragraphs_summary": {"1": "...", "2": "..."},
          "verbatim_excerpt": "High-risk AI systems which make use of...",
          "url": "...#art_10"
        }

    annex:
        same shape, with `article: null`, `annex: "III"`, `doc_type: "annex"`.

Run:
    python -m copilot.ingest.chunk_articles
    python -m copilot.ingest.chunk_articles --limit 3   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from copilot.llm import LLMClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "ai_act.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "articles.jsonl"

CORPUS_CELEX = "02024R1689-20260727"
BASE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"

# Closed vocabularies. Anything else returned by the LLM is dropped.
ROLES = {
    "provider",
    "deployer",
    "importer",
    "distributor",
    "authorised_representative",
    "product_manufacturer",
    "gpai_provider",
    "gpai_provider_systemic",
    "all",
}
TIERS = {
    "prohibited",
    "high_risk",
    "transparency",
    "gpai",
    "minimal",
    "all",
}

# Minimum length to keep an excerpt for ARTICLES. Below 40 chars, the
# paragraph is a placeholder like "Article 4" and carries no content.
# Annexes have no such placeholders, so their threshold is 0.
_MIN_EXCERPT_CHARS = 40
_EXCERPT_MAX_CHARS = 500


class ArticleMetadata(BaseModel):
    """LLM output for a single article/annex. Each field is validated."""

    roles: list[str] = Field(..., description="Operator roles the article applies to.")
    tiers: list[str] = Field(..., description="Risk tiers the article applies to.")
    summary: str = Field(..., description="2-4 sentence summary in plain English.")
    key_conditions: list[str] = Field(
        default_factory=list,
        description="0-6 snake_case conditions that trigger the article.",
    )
    paragraphs_summary: dict[str, str] = Field(
        default_factory=dict,
        description="Paragraph number -> one-line summary. Skip 'pu1'.",
    )


_SYSTEM_PROMPT = """You are a legal analyst specialised in the EU AI Act.

For the article or annex you receive, produce:

- `roles`: which operator roles it applies to. Allowed values:
  provider, deployer, importer, distributor, authorised_representative,
  product_manufacturer, gpai_provider, gpai_provider_systemic, all.
- `tiers`: which risk tiers it applies to. Allowed values:
  prohibited, high_risk, transparency, gpai, minimal, all.
- `summary`: 2-4 sentences in English describing the obligation concretely.
- `key_conditions`: 0-6 snake_case boolean conditions that trigger the
  article (e.g. "uses_training_data", "processes_personal_data",
  "is_public_body", "is_biometric", "annex_iii_point_4a").
  Empty for articles that apply unconditionally to their roles/tiers.
- `paragraphs_summary`: map paragraph number -> one-line English summary.
  Skip the unnumbered placeholder ("pu1", text is just "Article N").

Rules:
- Use ONLY the roles/tiers from the lists above. Never invent new values.
- Do not invent obligations. Stay close to the provided text.
- If unsure about a role/tier, prefer the broader one."""


def _excerpt(
    paragraphs: list[dict],
    *,
    min_chars: int = _MIN_EXCERPT_CHARS,
) -> str:
    """First paragraph with substantial content, truncated to 500 chars.

    Prefers numbered paragraphs because unnumbered ones (`pu1`) are often
    just "Article N" placeholders.

    `min_chars` is the threshold below which a paragraph is considered a
    placeholder. For articles it is 40 (filters "Article 4"). For annex
    blocks it is 0: any non-empty block is real content, even if short
    (e.g. "List of high-risk use cases.").
    """
    candidates = [p for p in paragraphs if (p.get("text") or "").strip()]
    numbered = [p for p in candidates if p.get("number") is not None]
    for p in numbered or candidates:
        text = p["text"].strip()
        if len(text) >= min_chars:
            if len(text) > _EXCERPT_MAX_CHARS:
                return text[:_EXCERPT_MAX_CHARS].rstrip() + "…"
            return text
    return ""


def _article_prompt(article: dict, chapter_title: str | None) -> str:
    lines = [
        f"Article {article['number']}: {article.get('title', '')}",
        f"Chapter: {chapter_title or '(none)'}",
        "",
        "Full text:",
    ]
    for p in article.get("paragraphs", []):
        text = (p.get("text") or "").strip()
        if text:
            lines.append(f"[p{p.get('number') or 'pu1'}] {text}")
    return "\n".join(lines)


def _annex_prompt(annex: dict) -> str:
    lines = [
        f"Annex {annex['number']}: {annex.get('title', '')}",
        "",
        "Full text:",
    ]
    for block in annex.get("blocks", []):
        text = (block.get("text") or "").strip()
        if text:
            lines.append(f"[b{block['n']}] {text}")
    return "\n".join(lines)


def _keep_only_known(values: list[str], allowed: set[str]) -> list[str]:
    """Drop hallucinated values; fall back to ['all'] if nothing survives."""
    kept = [v for v in values if v in allowed]
    return kept or ["all"]


def _build_record(
    *,
    record_id: str,
    doc_type: str,
    article: str | None,
    annex: str | None,
    title: str,
    url: str,
    excerpt: str,
    metadata: ArticleMetadata,
) -> dict:
    """Assemble the final dict, enforcing closed vocabularies."""
    return {
        "id": record_id,
        "doc_type": doc_type,
        "article": article,
        "annex": annex,
        "title": title,
        "roles": _keep_only_known(metadata.roles, ROLES),
        "tiers": _keep_only_known(metadata.tiers, TIERS),
        "summary": metadata.summary.strip(),
        "key_conditions": list(metadata.key_conditions),
        "paragraphs_summary": dict(metadata.paragraphs_summary),
        "verbatim_excerpt": excerpt,
        "url": url,
    }


def _process_article(
    article: dict, chapter_title: str | None, base_url: str, llm: LLMClient
) -> dict:
    metadata = llm.structured_complete(
        user_prompt=_article_prompt(article, chapter_title),
        schema=ArticleMetadata,
        system_prompt=_SYSTEM_PROMPT,
    )
    return _build_record(
        record_id=article["id"],
        doc_type="article",
        article=article["number"],
        annex=None,
        title=article.get("title", ""),
        url=f"{base_url}#art_{article['number']}",
        # Articles: keep the placeholder filter (default min_chars=40).
        excerpt=_excerpt(article.get("paragraphs", [])),
        metadata=metadata,
    )


def _process_annex(annex: dict, base_url: str, llm: LLMClient) -> dict:
    metadata = llm.structured_complete(
        user_prompt=_annex_prompt(annex),
        schema=ArticleMetadata,
        system_prompt=_SYSTEM_PROMPT,
    )
    return _build_record(
        record_id=annex["id"],
        doc_type="annex",
        article=None,
        annex=annex["number"],
        title=annex.get("title", ""),
        url=f"{base_url}#anx_{annex['number']}",
        # Annex blocks: no placeholder filter (min_chars=0). Any non-empty
        # block is real content, even when short.
        excerpt=_excerpt(annex.get("blocks", []), min_chars=0),
        metadata=metadata,
    )


def generate_article_records(doc: dict, *, llm: LLMClient, limit: int | None = None) -> list[dict]:
    """Produce one record per article and per annex. Stops at `limit` if set."""
    base_url = BASE_URL.format(celex=CORPUS_CELEX)
    records: list[dict] = []

    def _full() -> bool:
        return limit is not None and len(records) >= limit

    # Articles (chapter + section levels).
    for chapter in doc.get("chapters", []):
        for article in chapter.get("articles", []):
            if _full():
                return records
            records.append(_process_article(article, chapter.get("title"), base_url, llm))
        for section in chapter.get("sections", []):
            section_title = f"{chapter.get('title', '')} > {section.get('title', '')}"
            for article in section.get("articles", []):
                if _full():
                    return records
                records.append(_process_article(article, section_title, base_url, llm))

    # Annexes.
    for annex in doc.get("annexes", []):
        if _full():
            return records
        records.append(_process_annex(annex, base_url, llm))

    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N records (smoke test).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}. Run parse_html.py first.")

    doc = json.loads(args.input.read_text(encoding="utf-8"))
    llm = LLMClient()
    records = generate_article_records(doc, llm=llm, limit=args.limit)
    _write_jsonl(records, args.output)

    print(f"Articles: {len(records)}  ->  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
