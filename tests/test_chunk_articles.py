"""Tests for copilot.ingest.chunk_articles. No network (LLM is mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from copilot.ingest.chunk_articles import (
    ArticleMetadata,
    generate_article_records,
    main,
)


def _doc() -> dict:
    return {
        "chapters": [
            {
                "id": "cpt_I",
                "number": "I",
                "title": "GENERAL PROVISIONS",
                "articles": [
                    {
                        "id": "art_4",
                        "number": "4",
                        "base_number": 4,
                        "suffix": "",
                        "title": "AI literacy",
                        "paragraphs": [
                            {
                                "id": "art_4.pu1",
                                "number": None,
                                "text": "Article 4",
                                "amendments": [],
                                "mixed": False,
                            },
                            {
                                "id": "art_4.p1",
                                "number": "1",
                                "text": "Providers and deployers shall take measures to support AI literacy.",
                                "amendments": [],
                                "mixed": False,
                            },
                        ],
                        "amendments": [],
                        "mixed": False,
                    },
                ],
                "sections": [],
            },
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
                                        "id": "art_10.p1",
                                        "number": "1",
                                        "text": "High-risk AI systems using training data shall be developed on quality data.",
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
                        "heading": True,
                        "text": "List of high-risk use cases.",
                        "amendments": [],
                        "mixed": False,
                    },
                ],
            },
        ],
    }


def _mock_llm(metadata: ArticleMetadata) -> MagicMock:
    llm = MagicMock()
    llm.structured_complete.return_value = metadata
    return llm


def test_generates_one_record_per_article_and_annex():
    llm = _mock_llm(
        ArticleMetadata(
            roles=["provider"],
            tiers=["high_risk"],
            summary="x",
            key_conditions=[],
            paragraphs_summary={},
        )
    )
    records = generate_article_records(_doc(), llm=llm)
    assert [r["id"] for r in records] == ["art_4", "art_10", "anx_III"]


def test_article_record_schema():
    llm = _mock_llm(
        ArticleMetadata(
            roles=["provider"],
            tiers=["high_risk"],
            summary="Data governance obligations.",
            key_conditions=["uses_training_data"],
            paragraphs_summary={"1": "quality of training data"},
        )
    )
    records = generate_article_records(_doc(), llm=llm)
    art = next(r for r in records if r["id"] == "art_10")

    assert art["doc_type"] == "article"
    assert art["article"] == "10"
    assert art["annex"] is None
    assert art["title"] == "Data and data governance"
    assert art["roles"] == ["provider"]
    assert art["tiers"] == ["high_risk"]
    assert art["summary"] == "Data governance obligations."
    assert art["key_conditions"] == ["uses_training_data"]
    assert art["paragraphs_summary"] == {"1": "quality of training data"}
    assert art["verbatim_excerpt"].startswith("High-risk AI systems")
    assert art["url"].endswith("#art_10")


def test_annex_record_schema():
    llm = _mock_llm(
        ArticleMetadata(
            roles=["provider"],
            tiers=["high_risk"],
            summary="Annex III use cases.",
            key_conditions=[],
            paragraphs_summary={},
        )
    )
    records = generate_article_records(_doc(), llm=llm)
    anx = next(r for r in records if r["id"] == "anx_III")

    assert anx["doc_type"] == "annex"
    assert anx["article"] is None
    assert anx["annex"] == "III"
    assert anx["url"].endswith("#anx_III")
    assert anx["verbatim_excerpt"].startswith("List of high-risk")


def test_hallucinated_roles_and_tiers_are_dropped():
    llm = _mock_llm(
        ArticleMetadata(
            roles=["provider", "chief_ai_officer"],  # second one is not allowed
            tiers=["high_risk", "very_high"],  # second one is not allowed
            summary="x",
            key_conditions=[],
            paragraphs_summary={},
        )
    )
    records = generate_article_records(_doc(), llm=llm)
    for r in records:
        assert all(
            role
            in {
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
            for role in r["roles"]
        )
        assert all(
            tier
            in {
                "prohibited",
                "high_risk",
                "transparency",
                "gpai",
                "minimal",
                "all",
            }
            for tier in r["tiers"]
        )


def test_empty_roles_fall_back_to_all():
    llm = _mock_llm(
        ArticleMetadata(
            roles=[],
            tiers=[],  # nothing survives -> fallback ["all"]
            summary="x",
            key_conditions=[],
            paragraphs_summary={},
        )
    )
    records = generate_article_records(_doc(), llm=llm)
    for r in records:
        assert r["roles"] == ["all"]
        assert r["tiers"] == ["all"]


def test_limit_stops_early():
    llm = _mock_llm(
        ArticleMetadata(
            roles=["provider"],
            tiers=["high_risk"],
            summary="x",
            key_conditions=[],
            paragraphs_summary={},
        )
    )
    records = generate_article_records(_doc(), llm=llm, limit=1)
    assert len(records) == 1


def test_llm_is_called_once_per_record():
    llm = _mock_llm(
        ArticleMetadata(
            roles=["provider"],
            tiers=["high_risk"],
            summary="x",
            key_conditions=[],
            paragraphs_summary={},
        )
    )
    records = generate_article_records(_doc(), llm=llm)
    assert llm.structured_complete.call_count == len(records)


def test_main_writes_articles_jsonl(tmp_path: Path, monkeypatch):
    src = tmp_path / "ai_act.json"
    dst = tmp_path / "articles.jsonl"
    src.write_text(json.dumps(_doc()), encoding="utf-8")

    # Patch the LLMClient constructor so main() does not touch the network.
    fake_llm = _mock_llm(
        ArticleMetadata(
            roles=["provider"],
            tiers=["high_risk"],
            summary="x",
            key_conditions=[],
            paragraphs_summary={},
        )
    )
    monkeypatch.setattr(
        "copilot.ingest.chunk_articles.LLMClient",
        lambda: fake_llm,
    )

    rc = main(["--input", str(src), "--output", str(dst)])
    assert rc == 0
    assert dst.exists()
    lines = dst.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert {"id", "roles", "tiers", "summary", "url"} <= record.keys()
