"""Tests for indexing: only the payload builders are tested (no Qdrant)."""

from __future__ import annotations

from copilot.retrieval.index import _article_text, _paragraph_payload


def test_article_text_includes_summary_and_conditions():
    rec = {
        "id": "art_10",
        "title": "Data and data governance",
        "summary": "Data quality for high-risk systems.",
        "key_conditions": ["uses_training_data", "processes_personal_data"],
        "paragraphs_summary": {"1": "quality", "2": "governance"},
    }
    text = _article_text(rec)
    assert "Data and data governance" in text
    assert "Data quality" in text
    assert "uses_training_data" in text
    assert "p1: quality" in text


def test_paragraph_payload_keeps_chunk_id_and_text():
    rec = {
        "id": "art_10.p1",
        "doc_type": "article",
        "article": "10",
        "annex": None,
        "paragraph": "1",
        "chunk_id": "art_10.p1",
        "text": "Training data sets...",
        "article_title": "Data and data governance",
        "url": "http://x",
        "tokens": 100,
    }
    payload = _paragraph_payload(rec)
    assert payload["chunk_id"] == "art_10.p1"
    assert payload["text"].startswith("Training data")
