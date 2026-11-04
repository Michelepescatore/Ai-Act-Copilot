"""Typed state of the LangGraph graph."""

from __future__ import annotations

from typing import Literal, TypedDict

# Facts. Each fact has a status that says how reliable it is.
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
    user_input: str  # in English (already translated, if needed)

# Candidate articles selected by the article_discovery node.
class ArticleCandidate(TypedDict):
    id: str
    title: str
    score: float
    rationale: str

# Paragraphs retrieved from the corpus to support the synthesis.
class ParagraphEvidence(TypedDict):
    chunk_id: str
    article_id: str
    text: str
    score: float
    rank: int

# Synthesis produced by the model, with citations already verified against the corpus.
class Citation(TypedDict):
    claim: str
    chunk_id: str
    verified: bool


class Synthesis(TypedDict):
    summary: str
    citations: list[Citation]
    has_compliance_issue: bool

# Open questions: inferred, unknown or missing fields to clarify with the user.
class OpenQuestion(TypedDict):
    kind: Literal["inferred", "unknown", "missing"]
    field: str
    question: str

# Final report exposed to the caller.
class Report(TypedDict):
    summary: str
    facts: list[FactValue]
    articles: list[ArticleCandidate]
    paragraphs: list[ParagraphEvidence]
    citations: list[Citation]
    open_questions: list[OpenQuestion]
    unverified_citations: list[str]


class GraphState(TypedDict, total=False):
    # Original user input, in their language.
    user_input: str

    # Detected language and input translated into English.
    detected_language: str
    user_input_en: str
    final_answer: str

    # Pipeline internals.
    facts: CaseFacts
    candidate_articles: list[ArticleCandidate]
    paragraphs: list[ParagraphEvidence]
    synthesis: Synthesis

    # Clarification questions already reformulated (in English) and translated
    # into the user's language. Persisted between `prepare_questions` and
    # `ask_user` thanks to the checkpointer, so the LLM call is not repeated
    # on resume.
    pending_questions: list[str]

    # Flow control.
    needs_user_input: bool
    user_clarification: str
    user_clarification_original: str
    is_compliant: bool

    # Trace id to correlate the logs of the same execution.
    trace_id: str

    # True if the input contains suspicious prompt-injection patterns.
    suspicious_input: bool

    # Final output and error, if any.
    report: Report
    error: str | None