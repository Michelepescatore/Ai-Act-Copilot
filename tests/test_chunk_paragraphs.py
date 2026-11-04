"""Tests for copilot.ingest.chunk_paragraphs. No network, no tokenizer download."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from copilot.ingest.chunk_paragraphs import (
    generate_paragraph_chunks,
    main,
)


def _doc() -> dict:
    """Minimal doc: one article with p1 + p2 + a placeholder pu1, one annex with 2 blocks."""
    return {
        "chapters": [
            {
                "id": "cpt_III",
                "number": "III",
                "title": "HIGH-RISK AI SYSTEMS",
                "articles": [],
                "sections": [
                    {
                        "id": "cpt_III.sct_2",
                        "number": "2",
                        "title": "Requirements",
                        "articles": [
                            {
                                "id": "art_10",
                                "number": "10",
                                "base_number": 10,
                                "suffix": "",
                                "title": "Data and data governance",
                                "paragraphs": [
                                    {
                                        "id": "art_10.pu1",
                                        "number": None,
                                        "text": "Article 10",
                                        "amendments": [],
                                        "mixed": False,
                                    },
                                    {
                                        "id": "art_10.p1",
                                        "number": "1",
                                        "text": (
                                            "Training, validation and testing data sets "
                                            "shall be developed with quality data."
                                        ),
                                        "amendments": [],
                                        "mixed": False,
                                    },
                                    {
                                        "id": "art_10.p2",
                                        "number": "2",
                                        "text": (
                                            "Training, validation and testing data sets "
                                            "shall be subject to data governance."
                                        ),
                                        "amendments": [],
                                        "mixed": False,
                                    },
                                ],
                                "amendments": [],
                                "mixed": False,
                            },
                        ],
                    },
                ],
            },
        ],
        "annexes": [
            {
                "id": "anx_III",
                "number": "III",
                "title": "ANNEX III - High-risk AI systems",
                "blocks": [
                    {
                        "n": 1,
                        "heading": False,
                        "text": "Biometrics: remote identification.",
                        "amendments": [],
                        "mixed": False,
                    },
                    {
                        "n": 2,
                        "heading": False,
                        "text": "Critical infrastructure.",
                        "amendments": [],
                        "mixed": False,
                    },
                ],
            },
        ],
    }


def _fake_tokens(text: str) -> int:
    """Cheap tokenizer for tests: word count. Avoids bge-m3 download."""
    return len(text.split())


def test_skips_placeholder_paragraphs():
    with patch("copilot.ingest.chunk_paragraphs._count_tokens", _fake_tokens):
        chunks = generate_paragraph_chunks(_doc())
    ids = {c["id"] for c in chunks}
    assert "art_10.pu1" not in ids  # placeholder
    assert "art_10.p1" in ids
    assert "art_10.p2" in ids


def test_article_record_schema():
    with patch("copilot.ingest.chunk_paragraphs._count_tokens", _fake_tokens):
        chunks = generate_paragraph_chunks(_doc())
    art = next(c for c in chunks if c["id"] == "art_10.p1")

    assert art["doc_type"] == "article"
    assert art["article"] == "10"
    assert art["annex"] is None
    assert art["paragraph"] == "1"
    assert art["chunk_id"] == "art_10.p1"
    assert art["article_title"] == "Data and data governance"
    assert art["url"].endswith("#art_10")
    assert art["tokens"] > 0


def test_short_paragraph_has_no_suffix():
    # A paragraph under MAX_TOKENS stays as a single record, with no .sN suffix.
    with patch("copilot.ingest.chunk_paragraphs._count_tokens", _fake_tokens):
        chunks = generate_paragraph_chunks(_doc())
    art = next(c for c in chunks if c["chunk_id"] == "art_10.p1")
    assert art["text"].startswith("Training, validation")
    assert ".s" not in art["chunk_id"]


def test_annex_block_record_schema():
    with patch("copilot.ingest.chunk_paragraphs._count_tokens", _fake_tokens):
        chunks = generate_paragraph_chunks(_doc())
    blk = next(c for c in chunks if c["id"] == "anx_III.b1")

    assert blk["doc_type"] == "annex"
    assert blk["article"] is None
    assert blk["annex"] == "III"
    assert blk["paragraph"] is None
    assert blk["chunk_id"] == "anx_III.b1"
    assert blk["article_title"].startswith("ANNEX III")
    assert blk["url"].endswith("#anx_III")
    assert blk["tokens"] > 0


def test_skips_empty_text():
    doc = _doc()
    doc["annexes"][0]["blocks"].append(
        {"n": 3, "heading": False, "text": "   ", "amendments": [], "mixed": False}
    )
    with patch("copilot.ingest.chunk_paragraphs._count_tokens", _fake_tokens):
        chunks = generate_paragraph_chunks(doc)
    assert all(c["id"] != "anx_III.b3" for c in chunks)


def test_long_paragraph_kept_as_single_chunk():
    """I paragrafi lunghi NON vengono più spezzati: restano atomici.

    Sostituisce i vecchi `test_long_paragraph_is_split_and_keeps_id` e
    `test_split_preserves_all_text_content`, che verificavano uno split
    rimosso intenzionalmente (vedi la docstring di `chunk_paragraphs.py`:
    "Long paragraphs are NOT split").

    Il chunk_id coincide con l'id originale del paragrafo, senza suffissi
    `.s1`, `.s2`, ..., e il testo è preservato integralmente.
    """
    long_text = ("This is a sentence. " * 400).strip()
    doc = {
        "chapters": [
            {
                "id": "cpt_X",
                "number": "X",
                "title": "T",
                "sections": [],
                "articles": [
                    {
                        "id": "art_99",
                        "number": "99",
                        "title": "Long article",
                        "paragraphs": [
                            {
                                "id": "art_99.p1",
                                "number": "1",
                                "text": long_text,
                                "amendments": [],
                                "mixed": False,
                            },
                        ],
                        "amendments": [],
                        "mixed": False,
                    }
                ],
            }
        ],
        "annexes": [],
    }
    with patch("copilot.ingest.chunk_paragraphs._count_tokens", _fake_tokens):
        chunks = generate_paragraph_chunks(doc)

    # Un solo chunk, nonostante il testo sia molto lungo.
    assert len(chunks) == 1

    c = chunks[0]
    assert c["chunk_id"] == "art_99.p1"
    assert c["id"] == "art_99.p1"
    assert ".s" not in c["chunk_id"]
    assert c["article"] == "99"
    assert c["paragraph"] == "1"
    assert c["doc_type"] == "article"
    # Il testo è preservato integralmente (solo strip degli spazi ai bordi,
    # che qui non ci sono perché long_text è già `.strip()`-ato).
    assert c["text"] == long_text


def test_main_writes_paragraphs_jsonl(tmp_path: Path):
    src = tmp_path / "ai_act.json"
    dst = tmp_path / "paragraphs.jsonl"
    src.write_text(json.dumps(_doc()), encoding="utf-8")

    with patch("copilot.ingest.chunk_paragraphs._count_tokens", _fake_tokens):
        rc = main(["--input", str(src), "--output", str(dst)])

    assert rc == 0
    assert dst.exists()
    lines = dst.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # 2 paragraphs + 2 annex blocks
    for line in lines:
        record = json.loads(line)
        assert "chunk_id" in record and "text" in record and "tokens" in record