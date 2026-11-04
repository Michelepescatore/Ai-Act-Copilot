"""Stato tipizzato del grafo LangGraph."""

from __future__ import annotations

from typing import Literal, TypedDict

# Fatti. Ogni fatto ha uno stato che dice quanto è affidabile.
FactStatus = Literal["known", "inferred", "unknown", "missing"]


class FactValue(TypedDict):
    field: str
    value: str | None
    status: FactStatus
    evidence: str
    rationale: str
    required: bool


class CaseFacts(TypedDict):
    facts: list[FactValue]
    user_input: str  # in inglese (già tradotto, se necessario)

# Articoli candidati selezionati dal nodo article_discovery.
class ArticleCandidate(TypedDict):
    id: str
    title: str
    score: float
    rationale: str

# Paragrafi recuperati dal corpus per supportare la sintesi.
class ParagraphEvidence(TypedDict):
    chunk_id: str
    article_id: str
    text: str
    score: float
    rank: int

# Sintesi prodotta dal modello, con citazioni già verificate contro il corpus.
class Citation(TypedDict):
    claim: str
    chunk_id: str
    verified: bool


class Synthesis(TypedDict):
    summary: str
    citations: list[Citation]
    has_compliance_issue: bool

# Domande aperte: campi inferiti, sconosciuti o mancanti da chiarire con l'utente.
class OpenQuestion(TypedDict):
    kind: Literal["inferred", "unknown", "missing"]
    field: str
    question: str

# Report finale esposto al chiamante.
class Report(TypedDict):
    summary: str
    facts: list[FactValue]
    articles: list[ArticleCandidate]
    paragraphs: list[ParagraphEvidence]
    citations: list[Citation]
    open_questions: list[OpenQuestion]
    unverified_citations: list[str]


class GraphState(TypedDict, total=False):
    # Input originale dell'utente, nella sua lingua.
    user_input: str

    # Lingua rilevata e input tradotto in inglese.
    detected_language: str
    user_input_en: str
    final_answer: str

    # Interni della pipeline.
    facts: CaseFacts
    candidate_articles: list[ArticleCandidate]
    paragraphs: list[ParagraphEvidence]
    synthesis: Synthesis

    # Domande di chiarificazione già riformulate (in inglese) e tradotte
    # nella lingua utente. Persistenti tra `prepare_questions` e `ask_user`
    # grazie al checkpointer, così al resume non si rifà la chiamata LLM.
    pending_questions: list[str]

    # Controllo di flusso.
    needs_user_input: bool
    user_clarification: str
    user_clarification_original: str
    is_compliant: bool

    # Trace id per correlare i log della stessa esecuzione.
    trace_id: str

    # True se l'input contiene pattern sospetti di prompt injection.
    suspicious_input: bool

    # Output finale e errore, se presente.
    report: Report
    error: str | None