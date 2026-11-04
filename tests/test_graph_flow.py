"""Test end-to-end del grafo LangGraph."""

from __future__ import annotations

from langgraph.types import Command

from copilot.graph.graph import compile_graph
from copilot.llm import LLMError


def _hits():
    return {
        "articles_v1": [
            {
                "id": "art_10",
                "score": 0.9,
                "payload": {"title": "Data governance", "summary": "Obligations."},
            }
        ],
        "paragraphs_v1": [
            {
                "id": "art_10.p1",
                "score": 0.9,
                "payload": {"article": "10", "text": "Training data shall be..."},
            }
        ],
    }


class _HitsRetriever:
    def search(self, query, *, collection, method="hybrid", limit=10):
        return _hits().get(collection, [])[:limit]


def test_happy_path_english_compliant(fake_llm, patch_retriever):
    patch_retriever(_HitsRetriever())
    llm = fake_llm(
        [
            {"code": "en"},
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": "deployer",
                        "status": "known",
                        "evidence": "we deploy",
                        "rationale": "Explicit.",
                        "required": True,
                    }
                ]
            },
            {"articles": [{"id": "art_10", "rationale": "Governance."}]},
            {
                "summary": "This system is compliant because it is not high-risk.",
                "citations": [{"claim": "Not high-risk.", "chunk_id": "art_10.p1"}],
                "has_compliance_issue": False,
            },
        ]
    )
    app = compile_graph(llm=llm)
    config = {"configurable": {"thread_id": "t1"}}
    result = app.invoke({"user_input": "We deploy a spam filter."}, config=config)
    assert "__interrupt__" not in result
    assert "compliant" in result["report"]["summary"].lower()
    assert result["report"]["citations"][0]["verified"] is True
    assert len(llm.calls) == 4


def test_happy_path_english_noncompliant(fake_llm, patch_retriever):
    patch_retriever(_HitsRetriever())
    llm = fake_llm(
        [
            {"code": "en"},
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": "provider",
                        "status": "known",
                        "evidence": "we provide",
                        "rationale": "Explicit.",
                        "required": True,
                    }
                ]
            },
            {"articles": [{"id": "art_10", "rationale": "Governance."}]},
            {
                "summary": "There are compliance issues.",
                "citations": [{"claim": "Data governance applies.", "chunk_id": "art_10.p1"}],
                "has_compliance_issue": True,
            },
        ]
    )
    app = compile_graph(llm=llm)
    config = {"configurable": {"thread_id": "t2"}}
    result = app.invoke({"user_input": "We provide a hiring system."}, config=config)
    assert "compliance" in result["report"]["summary"].lower()
    assert result["report"]["citations"][0]["verified"] is True


def test_multilingual_italian_end_to_end(fake_llm, patch_retriever):
    patch_retriever(_HitsRetriever())
    llm = fake_llm(
        [
            {"code": "it"},
            {"text": "We provide an AI system for hiring."},
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": "provider",
                        "status": "known",
                        "evidence": "We provide",
                        "rationale": "Explicit.",
                        "required": True,
                    }
                ]
            },
            {"articles": [{"id": "art_10", "rationale": "Governance."}]},
            {
                "summary": "You are a provider.",
                "citations": [{"claim": "Obligations.", "chunk_id": "art_10.p1"}],
                "has_compliance_issue": True,
            },
            {"text": "Sei un fornitore."},
        ]
    )
    app = compile_graph(llm=llm)
    config = {"configurable": {"thread_id": "t3"}}
    result = app.invoke({"user_input": "Forniamo un sistema AI."}, config=config)
    assert result["detected_language"] == "it"
    assert result["user_input_en"] == "We provide an AI system for hiring."
    assert result["report"]["summary"] == "Sei un fornitore."
    assert len(llm.calls) == 6


def test_unsupported_language_falls_back(fake_llm, patch_retriever):
    patch_retriever(_HitsRetriever())
    llm = fake_llm(
        [
            {"code": "xx"},
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": "provider",
                        "status": "known",
                        "evidence": "",
                        "rationale": "Explicit.",
                        "required": True,
                    }
                ]
            },
            {"articles": [{"id": "art_10", "rationale": "Governance."}]},
            {"summary": "OK.", "citations": [], "has_compliance_issue": False},
        ]
    )
    app = compile_graph(llm=llm)
    config = {"configurable": {"thread_id": "t4"}}
    result = app.invoke({"user_input": "Some text."}, config=config)
    assert result["detected_language"] == "en"
    assert result["user_input_en"] == "Some text."
    assert len(llm.calls) == 4


def test_clarification_loop_italian(fake_llm, patch_retriever):
    patch_retriever(_HitsRetriever())
    llm = fake_llm(
        [
            # 1) language_detect
            {"code": "it"},
            # 2) translate_input
            {"text": "We provide a system for candidate evaluation."},
            # 3) fact_extraction (primo giro): operator_role mancante
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": None,
                        "status": "missing",
                        "evidence": "",
                        "rationale": "The text does not specify your role.",
                        "required": True,
                    },
                ]
            },
            # 4) prepare_questions: riformulazione in linguaggio naturale
            {"questions": ["What is your role for this AI system?"]},
            # 5) prepare_questions: traduzione della domanda in italiano
            {"text": "Qual è il tuo ruolo per questo sistema di IA?"},
            # 6) fact_extraction (secondo giro): traduzione del chiarimento
            {"text": "I am the provider."},
            # 7) fact_extraction (secondo giro): estrazione aggiornata
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": "provider",
                        "status": "known",
                        "evidence": "I am the provider",
                        "rationale": "User confirmed.",
                        "required": True,
                    },
                    {
                        "field": "risk_tier",
                        "value": "high_risk",
                        "status": "inferred",
                        "evidence": "candidate evaluation",
                        "rationale": "Annex III hiring.",
                        "required": False,
                    },
                ]
            },
            # 8) article_discovery
            {"articles": [{"id": "art_10", "rationale": "Governance."}]},
            # 9) synthesis
            {
                "summary": "High-risk system.",
                "citations": [{"claim": "Obligations.", "chunk_id": "art_10.p1"}],
                "has_compliance_issue": True,
            },
            # 10) translate_output: traduzione della summary
            {"text": "Sistema ad alto rischio."},
            # 11) translate_output: traduzione della domanda aperta su risk_tier
            {"text": "Ho assunto che risk_tier = high_risk. Confermi?"},
        ]
    )
    app = compile_graph(llm=llm)
    config = {"configurable": {"thread_id": "t5"}}
    result = app.invoke({"user_input": "Forniamo un sistema per la valutazione."}, config=config)

    # Interrupt atteso: manca operator_role
    assert "__interrupt__" in result
    questions = result["__interrupt__"][0].value["questions"]
    # La domanda mostrata all'utente è in italiano (riformulata + tradotta)
    assert questions == ["Qual è il tuo ruolo per questo sistema di IA?"]

    # Riprendi con il chiarimento (in italiano)
    result = app.invoke(Command(resume="Sono il fornitore."), config=config)
    assert "__interrupt__" not in result
    assert result["user_clarification_original"] == "Sono il fornitore."
    assert "provider" in result["user_clarification"].lower()
    role = next(f for f in result["facts"]["facts"] if f["field"] == "operator_role")
    assert role["status"] == "known"
    assert role["value"] == "provider"
    # La domanda aperta su risk_tier è stata tradotta in italiano
    open_questions = result["report"]["open_questions"]
    assert any("risk_tier" in q["field"] for q in open_questions)
    assert any("Confermi" in q["question"] for q in open_questions)


def test_llm_error_routes_to_error_node(fake_llm, patch_retriever):
    patch_retriever(_HitsRetriever())
    llm = fake_llm([{"code": "en"}, LLMError("boom")])
    app = compile_graph(llm=llm)
    config = {"configurable": {"thread_id": "t6"}}
    result = app.invoke({"user_input": "Hello."}, config=config)
    assert result.get("error", "").startswith("fact_extraction")
    assert "errore" in result["report"]["summary"].lower()
    assert result["report"]["citations"] == []
