"""Unit tests for the graph nodes."""

from __future__ import annotations

from copilot.graph import nodes
from copilot.graph.state import CaseFacts, FactValue, GraphState
from copilot.llm import LLMError


def _make_fact(field, value, status, required=False, rationale=""):
    return FactValue(
        field=field,
        value=value,
        status=status,
        evidence="",
        rationale=rationale or f"{field} = {value}",
        required=required,
    )


# --- fact_extraction ---------------------------------------------------------

def test_fact_extraction_translates_italian_clarification(fake_llm):
    """With an Italian clarification, fact_extraction makes 2 calls:
    1) translation (translate_to_english)
    2) fact extraction (_FactExtractionLLM)."""
    fake = fake_llm(
        [
            {"text": "Yes, the system evaluates candidates for job offers."},
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": "provider",
                        "status": "known",
                        "evidence": "forniamo un sistema",
                        "rationale": "L'utente dice di fornire il sistema.",
                        "required": True,
                    }
                ]
            },
        ]
    )
    state: GraphState = {
        "user_input": "Siamo un'azienda che fornisce un sistema.",
        "user_input_en": "We are a company that provides a system.",
        "detected_language": "it",
        "user_clarification_original": "Sì, il sistema valuta i candidati.",
    }
    result = nodes.fact_extraction(state, llm=fake)

    assert len(fake.calls) == 2
    # The first call is the translation (translate_to_english system prompt).
    assert "Traduci il testo dell'utente in inglese" in fake.calls[0]["system_prompt"]
    # The second is the fact extraction.
    assert fake.calls[1]["schema"] == "_FactExtractionLLM"
    assert "job offers" in fake.calls[1]["user_prompt"]
    # The English clarification was stored.
    assert result["user_clarification"] == (
        "Yes, the system evaluates candidates for job offers."
    )
    assert result["needs_user_input"] is False


def test_fact_extraction_english_clarification_not_translated(fake_llm):
    """If the language is 'en', the clarification is not translated: 1 call only."""
    fake = fake_llm(
        [
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": "provider",
                        "status": "known",
                        "evidence": "we provide",
                        "rationale": "explicit",
                        "required": True,
                    }
                ]
            }
        ]
    )
    state: GraphState = {
        "user_input": "We provide an AI system.",
        "user_input_en": "We provide an AI system.",
        "detected_language": "en",
        "user_clarification_original": "Yes, we are the provider.",
    }
    result = nodes.fact_extraction(state, llm=fake)
    assert len(fake.calls) == 1
    assert result["user_clarification"] == "Yes, we are the provider."


def test_fact_extraction_blocking_sets_needs_user_input(fake_llm):
    fake = fake_llm(
        [
            {
                "facts": [
                    {
                        "field": "risk_tier",
                        "value": None,
                        "status": "missing",
                        "evidence": "",
                        "rationale": "Not specified",
                        "required": True,
                    }
                ]
            }
        ]
    )
    state: GraphState = {
        "user_input": "A case.",
        "user_input_en": "A case.",
        "detected_language": "en",
    }
    result = nodes.fact_extraction(state, llm=fake)
    assert result["needs_user_input"] is True


def test_fact_extraction_llm_error_returns_error_state(fake_llm):
    fake = fake_llm([LLMError("boom")])
    state: GraphState = {
        "user_input": "x",
        "user_input_en": "x",
        "detected_language": "en",
    }
    result = nodes.fact_extraction(state, llm=fake)
    assert "error" in result
    assert "fact_extraction" in result["error"]


# --- article_discovery -------------------------------------------------------

def test_article_discovery_filters_unknown_ids(fake_llm, patch_retriever):
    class R:
        def search(self, q, *, collection, method, limit):
            return [
                {"id": "art_10", "score": 0.9, "payload": {"title": "Data governance"}}
            ]

    patch_retriever(R())
    fake = fake_llm(
        [
            {
                "articles": [
                    {"id": "art_10", "rationale": "Data governance."},
                    {"id": "art_999", "rationale": "Invented."},
                ]
            }
        ]
    )
    state: GraphState = {
        "facts": CaseFacts(
            facts=[_make_fact("operator_role", "provider", "known")],
            user_input="case",
        ),
        "detected_language": "en",
    }
    result = nodes.article_discovery(state, llm=fake)
    ids = [a["id"] for a in result["candidate_articles"]]
    assert ids == ["art_10"]


def test_article_discovery_uses_hybrid_rerank(fake_llm, patch_retriever):
    calls = []

    class R:
        def search(self, q, *, collection, method, limit):
            calls.append({"collection": collection, "method": method})
            return []

    patch_retriever(R())
    fake = fake_llm([{"articles": []}])
    state: GraphState = {
        "facts": CaseFacts(
            facts=[_make_fact("operator_role", "provider", "known")],
            user_input="case",
        ),
        "detected_language": "en",
    }
    nodes.article_discovery(state, llm=fake)
    assert calls[0]["collection"] == "articles_v1"
    assert calls[0]["method"] == "hybrid_rerank"


# --- paragraph_check ---------------------------------------------------------

def test_paragraph_check_dedup_and_filter(patch_retriever):
    class R:
        def search(self, q, *, collection, method, limit):
            return [
                {"id": "art_10.p1", "score": 0.9, "payload": {"article": "10", "text": "a"}},
                {"id": "art_11.p1", "score": 0.8, "payload": {"article": "11", "text": "b"}},
            ]

    patch_retriever(R())
    state: GraphState = {
        "facts": CaseFacts(
            facts=[_make_fact("operator_role", "provider", "known")],
            user_input="case",
        ),
        "candidate_articles": [
            {"id": "art_10", "title": "T", "score": 0.9, "rationale": "r"}
        ],
        "detected_language": "en",
    }
    result = nodes.paragraph_check(state)
    assert [p["chunk_id"] for p in result["paragraphs"]] == ["art_10.p1"]


def test_paragraph_check_fallback_when_no_match(patch_retriever):
    class R:
        def search(self, q, *, collection, method, limit):
            return [
                {"id": "art_99.p1", "score": 0.5, "payload": {"article": "99", "text": "x"}},
                {"id": "art_99.p2", "score": 0.4, "payload": {"article": "99", "text": "y"}},
            ]

    patch_retriever(R())
    state: GraphState = {
        "facts": CaseFacts(
            facts=[_make_fact("operator_role", "provider", "known")],
            user_input="case",
        ),
        "candidate_articles": [
            {"id": "art_10", "title": "T", "score": 0.9, "rationale": "r"}
        ],
        "detected_language": "en",
    }
    result = nodes.paragraph_check(state)
    assert len(result["paragraphs"]) == 2


# --- synthesis ---------------------------------------------------------------

def test_synthesis_verifies_citations(fake_llm):
    fake = fake_llm(
        [
            {
                "summary": "Summary.",
                "citations": [
                    {"claim": "A", "chunk_id": "art_10.p1"},
                    {"claim": "B", "chunk_id": "art_99.p9"},
                ],
                "has_compliance_issue": True,
            }
        ]
    )
    state: GraphState = {
        "facts": CaseFacts(
            facts=[_make_fact("operator_role", "provider", "known")],
            user_input="case",
        ),
        "candidate_articles": [],
        "paragraphs": [
            {"chunk_id": "art_10.p1", "article_id": "art_10", "text": "t", "score": 0.9, "rank": 1}
        ],
        "detected_language": "en",
    }
    result = nodes.synthesis(state, llm=fake)
    cits = result["synthesis"]["citations"]
    assert cits[0]["verified"] is True
    assert cits[1]["verified"] is False
    assert result["is_compliant"] is False


def test_synthesis_llm_error(fake_llm):
    fake = fake_llm([LLMError("boom")])
    state: GraphState = {
        "facts": CaseFacts(facts=[], user_input="x"),
        "paragraphs": [],
        "candidate_articles": [],
        "detected_language": "en",
    }
    result = nodes.synthesis(state, llm=fake)
    assert "error" in result


# --- report_node / no_compliance_issue / error_node -------------------------

def test_report_node_open_questions_grouped():
    state: GraphState = {
        "facts": CaseFacts(
            facts=[
                _make_fact("a", "v", "inferred"),
                _make_fact("b", None, "unknown"),
                _make_fact("c", None, "missing"),
            ],
            user_input="x",
        ),
        "candidate_articles": [],
        "paragraphs": [],
        "synthesis": {
            "summary": "S",
            "citations": [
                {"claim": "x", "chunk_id": "art_1.p1", "verified": True},
                {"claim": "y", "chunk_id": "art_2.p1", "verified": False},
            ],
            "has_compliance_issue": True,
        },
        "detected_language": "en",
    }
    result = nodes.report_node(state)
    report = result["report"]
    assert [q["kind"] for q in report["open_questions"]] == [
        "inferred",
        "unknown",
        "missing",
    ]
    assert report["unverified_citations"] == ["art_2.p1"]


def test_no_compliance_issue_has_no_questions():
    state: GraphState = {
        "facts": CaseFacts(facts=[_make_fact("a", None, "unknown")], user_input="x"),
        "candidate_articles": [],
        "paragraphs": [],
        "synthesis": {"summary": "Compliant.", "citations": [], "has_compliance_issue": False},
        "detected_language": "en",
    }
    result = nodes.no_compliance_issue(state)
    assert result["report"]["open_questions"] == []


def test_error_node_produces_minimal_report():
    state: GraphState = {
        "error": "synthesis: boom",
        "facts": CaseFacts(facts=[], user_input=""),
    }
    result = nodes.error_node(state)
    assert "errore" in result["report"]["summary"].lower()
    assert result["report"]["citations"] == []