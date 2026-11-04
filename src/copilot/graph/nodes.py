"""LangGraph graph nodes."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from copilot.graph.prompts import (
    ARTICLE_DISCOVERY_SYSTEM,
    FACT_EXTRACTION_SYSTEM,
    QUESTION_REFORMULATION_SYSTEM,
    REPORT_DISCLAIMER,
    SYNTHESIS_SYSTEM,
)
from copilot.graph.state import (
    ArticleCandidate,
    CaseFacts,
    Citation,
    FactValue,
    GraphState,
    OpenQuestion,
    ParagraphEvidence,
    Report,
    Synthesis,
)
from copilot.llm import LLMClient, LLMError
from copilot.retrieval.search import Retriever
from copilot.retrieval.store import QdrantStore

logger = logging.getLogger(__name__)

# Retrieval method used by the graph.
RETRIEVAL_METHOD = "hybrid_rerank"

# Maximum length of the English input. Beyond this, we reject with a handled error.
MAX_INPUT_CHARS = 4000

# Allowed values for fields with a closed vocabulary. Used to suggest options
# in the clarification question.
_ALLOWED_VALUES: dict[str, list[str]] = {
    "operator_role": [
        "provider",
        "deployer",
        "importer",
        "distributor",
        "authorised_representative",
        "product_manufacturer",
        "gpai_provider",
        "gpai_provider_systemic",
    ],
}

# Prompt-injection patterns. We do not block, but we flag them.
_INJECTION_PATTERNS = [
    re.compile(r"\bignore (all |any )?(previous |prior )?instructions\b", re.I),
    re.compile(r"\bignore (the )?(system|prompt|rules)\b", re.I),
    re.compile(r"\bdisregard (all |any )?(previous |prior )?instructions\b", re.I),
    re.compile(r"\bforget (all |any )?(previous |prior )?instructions\b", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
]

# Forbidden phrases in the output. If the model produces them, they are replaced.
FORBIDDEN_PATTERNS = [
    re.compile(r"\byour system is (fully |pienamente )?compliant\b", re.I),
    re.compile(r"\byou are (fully |pienamente )?compliant\b", re.I),
    re.compile(r"\byou have nothing (more )?to do\b", re.I),
    re.compile(r"\bno obligations? appl(y|ies)\b", re.I),
    re.compile(r"\b(certainly|definitely) compliant\b", re.I),
]

_NEUTRAL_FALLBACK = (
    "Some obligations may apply to your system. "
    "Please verify the specific requirements with a qualified professional."
)


class _FactValueLLM(BaseModel):
    field: str
    value: str | None = None
    status: str = Field(..., pattern="^(known|inferred|unknown|missing)$")
    evidence: str = ""
    rationale: str
    required: bool = False


class _FactExtractionLLM(BaseModel):
    facts: list[_FactValueLLM]


class _ArticleCandidateLLM(BaseModel):
    id: str
    rationale: str


class _ArticleDiscoveryLLM(BaseModel):
    articles: list[_ArticleCandidateLLM]


class _CitationLLM(BaseModel):
    claim: str
    chunk_id: str


class _SynthesisLLM(BaseModel):
    summary: str
    citations: list[_CitationLLM]
    has_compliance_issue: bool


# Schema for the reformulation of clarification questions.
class _QuestionReformulationLLM(BaseModel):
    questions: list[str]


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    """Retriever singleton. Tests patch it via cache_clear."""
    from copilot.retrieval.embedders import Embedders

    return Retriever(
        embedders=Embedders(),
        store=QdrantStore(url="http://localhost:6333"),
    )


def _query_from_facts(facts: CaseFacts) -> str:
    """Build the retrieval query from the known or inferred facts."""
    parts = [
        f"{f['field']}={f['value']}"
        for f in facts["facts"]
        if f["status"] in ("known", "inferred") and f["value"]
    ]
    return " ".join(parts) if parts else facts["user_input"]


def _has_suspicious_pattern(text: str) -> bool:
    """True if the text contains a known prompt-injection pattern."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def _neutralize_forbidden(text: str) -> str:
    """Replace forbidden phrases with a neutral formulation."""
    result = text
    for pat in FORBIDDEN_PATTERNS:
        result, n = pat.subn(_NEUTRAL_FALLBACK, result)
        if n:
            logger.warning("forbidden phrase neutralized: %s", pat.pattern)
    return result


def _apply_report_guards(report: dict) -> dict:
    """Neutralize forbidden phrases, add disclaimer and unverified-citation warning."""
    summary = report.get("summary", "")
    summary = _neutralize_forbidden(summary)

    unverified = report.get("unverified_citations", [])
    if unverified:
        summary += (
            f"\n\nWarning: {len(unverified)} citation(s) could not be verified "
            "against the AI Act corpus. Treat the analysis with caution."
        )

    summary = summary.rstrip() + "\n\n" + REPORT_DISCLAIMER
    return {**report, "summary": summary}


def fact_extraction(state: GraphState, *, llm: LLMClient) -> dict[str, Any]:
    """Extract the facts from the case (in English, after translation)."""
    from copilot.graph.language import translate_to_english

    user_input_en = state.get("user_input_en") or state.get("user_input", "")
    clarification_original = state.get("user_clarification_original")
    code = state.get("detected_language", "en")

    # Reject inputs that are too long before calling the model.
    if len(user_input_en) > MAX_INPUT_CHARS:
        return {"error": f"input_too_long: {len(user_input_en)} > {MAX_INPUT_CHARS}"}

    # Detect possible prompt injection. Does not block: proceeds and flags.
    suspicious = _has_suspicious_pattern(user_input_en)
    if suspicious:
        logger.warning("suspicious input detected (prompt injection pattern)")

    # Translate the user's clarification (if any) into English.
    clarification_en = ""
    if clarification_original:
        if code != "en":
            clarification_en = translate_to_english(clarification_original, llm)
        else:
            clarification_en = clarification_original

    # Build the prompt with the input delimited by tags.
    parts = [f"<user_input>\n{user_input_en}\n</user_input>"]
    if clarification_en:
        parts.append(f"<user_clarification>\n{clarification_en}\n</user_clarification>")
    prompt = "\n\n".join(parts)

    try:
        result = llm.structured_complete(
            prompt,
            _FactExtractionLLM,
            system_prompt=FACT_EXTRACTION_SYSTEM,
        )
    except LLMError as exc:
        logger.exception("fact_extraction failed")
        return {"error": f"fact_extraction: {exc}"}

    facts = [
        FactValue(
            field=f.field,
            value=f.value,
            status=f.status,  # type: ignore[arg-type]
            evidence=f.evidence,
            rationale=f.rationale,
            required=f.required,
        )
        for f in result.facts
    ]

    # Discard non-required "unknown" facts: the prompt should already
    # omit them, this is a safety net.
    facts = [f for f in facts if f["status"] != "unknown" or f["required"]]

    blocking = [
        f for f in facts
        if f["required"] and f["status"] in ("unknown", "missing")
    ]

    return {
        "facts": CaseFacts(facts=facts, user_input=user_input_en),
        "needs_user_input": bool(blocking),
        "user_clarification": clarification_en,
        "suspicious_input": suspicious,
    }


def ask_user(state: GraphState, *, llm: LLMClient) -> dict[str, Any]:
    """Rephrase the questions for the user in natural language.

    Takes the missing required facts, builds a prompt with field,
    rationale and allowed values (if any), and asks the LLM to produce
    a question readable by a non-expert user. If the call fails,
    it falls back to a simple template.
    """
    facts = state["facts"]["facts"]
    blocking = [
        f for f in facts
        if f["required"] and f["status"] in ("unknown", "missing")
    ]
    if not blocking:
        return {"questions": []}

    # Build the description of the missing facts for the prompt.
    lines: list[str] = []
    for f in blocking:
        allowed = _ALLOWED_VALUES.get(f["field"], [])
        lines.append(
            f"- field: {f['field']}\n"
            f"  rationale: {f['rationale']}\n"
            f"  allowed_values: {allowed}"
        )
    prompt = "Missing facts:\n\n" + "\n\n".join(lines)

    try:
        result = llm.structured_complete(
            prompt,
            _QuestionReformulationLLM,
            system_prompt=QUESTION_REFORMULATION_SYSTEM,
        )
        questions = result.questions
    except LLMError as exc:
        # Fallback: simple template, consistent with the number of facts.
        logger.warning("ask_user reformulation failed: %s", exc)
        questions = [
            f"Please provide a value for `{f['field']}`." for f in blocking
        ]

    return {"questions": questions}  # type: ignore[typeddict-unknown-key]


def article_discovery(state: GraphState, *, llm: LLMClient) -> dict[str, Any]:
    """Select the candidate articles via retrieval + LLM."""
    if state.get("error"):
        return {}

    facts = state["facts"]
    query = _query_from_facts(facts)

    hits = _retriever().search(
        query,
        collection="articles_v1",
        method=RETRIEVAL_METHOD,
        limit=15,
    )

    candidates_text = "\n".join(
        f"- id={h['id']} | title={h['payload'].get('title','')} | "
        f"summary={h['payload'].get('summary','')[:200]}"
        for h in hits
    )
    prompt = (
        f"Case facts:\n{query}\n\n"
        f"Retrieved candidate articles:\n{candidates_text}"
    )

    try:
        result = llm.structured_complete(
            prompt,
            _ArticleDiscoveryLLM,
            system_prompt=ARTICLE_DISCOVERY_SYSTEM,
        )
    except LLMError as exc:
        logger.exception("article_discovery failed")
        return {"error": f"article_discovery: {exc}"}

    by_id = {h["id"]: h for h in hits}
    candidates: list[ArticleCandidate] = []
    for c in result.articles:
        if c.id in by_id:
            candidates.append(
                ArticleCandidate(
                    id=c.id,
                    title=by_id[c.id]["payload"].get("title", ""),
                    score=by_id[c.id]["score"],
                    rationale=c.rationale,
                )
            )

    return {"candidate_articles": candidates}


def paragraph_check(state: GraphState) -> dict[str, Any]:
    """Retrieve the relevant paragraphs in a single pass, then filter by article.

    Optimization: previously we made one retrieval call per candidate article.
    Now we make a single call with a higher limit, then filter by article
    in memory.
    """
    if state.get("error"):
        return {}

    facts = state["facts"]
    query = _query_from_facts(facts)
    candidate_articles = state.get("candidate_articles", [])
    if not candidate_articles:
        return {"paragraphs": []}

    hits = _retriever().search(
        query,
        collection="paragraphs_v1",
        method=RETRIEVAL_METHOD,
        limit=50,
    )

    # Index hits by article number, for O(1) filtering.
    by_article: dict[str, list[dict]] = {}
    for h in hits:
        article_num = str(h["payload"].get("article") or "")
        if not article_num:
            continue
        by_article.setdefault(article_num, []).append(h)

    all_paragraphs: list[ParagraphEvidence] = []
    seen: set[str] = set()

    for article in candidate_articles:
        article_num = article["id"].replace("art_", "")
        article_hits = by_article.get(article_num, [])

        # Fallback: if the article has no direct hits, take the first 3 global ones.
        if not article_hits:
            article_hits = hits[:3]

        for rank, h in enumerate(article_hits[:5], start=1):
            chunk_id = h["id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            all_paragraphs.append(
                ParagraphEvidence(
                    chunk_id=chunk_id,
                    article_id=article["id"],
                    text=h["payload"].get("text", ""),
                    score=h["score"],
                    rank=rank,
                )
            )

    return {"paragraphs": all_paragraphs}


def synthesis(state: GraphState, *, llm: LLMClient) -> dict[str, Any]:
    """Produce the synthesis and verify the citations against the context."""
    if state.get("error"):
        return {}

    facts = state["facts"]
    facts_text = "\n".join(
        f"- {f['field']}: {f['value']} (status={f['status']}, {f['rationale']})"
        for f in facts["facts"]
    )
    articles_text = "\n".join(
        f"- {a['id']}: {a['title']} — {a['rationale']}"
        for a in state.get("candidate_articles", [])
    )
    paragraphs_text = "\n\n".join(
        f"[{p['chunk_id']}] (article={p['article_id']})\n{p['text']}"
        for p in state.get("paragraphs", [])
    )

    prompt = (
        f"Case facts:\n{facts_text}\n\n"
        f"Candidate articles:\n{articles_text}\n\n"
        f"Relevant paragraphs:\n{paragraphs_text}\n\n"
        f"Original question: {facts['user_input']}"
    )

    try:
        result = llm.structured_complete(
            prompt,
            _SynthesisLLM,
            system_prompt=SYNTHESIS_SYSTEM,
        )
    except LLMError as exc:
        logger.exception("synthesis failed")
        return {"error": f"synthesis: {exc}"}

    known_paragraph_chunks = {p["chunk_id"] for p in state.get("paragraphs", [])}
    known_article_ids = {a["id"] for a in state.get("candidate_articles", [])}
    known_chunks = known_paragraph_chunks | known_article_ids

    citations = [
        Citation(
            claim=c.claim,
            chunk_id=c.chunk_id,
            verified=c.chunk_id in known_chunks,
        )
        for c in result.citations
    ]

    return {
        "synthesis": Synthesis(
            summary=result.summary,
            citations=citations,
            has_compliance_issue=result.has_compliance_issue,
        ),
        "is_compliant": not result.has_compliance_issue,
    }


def no_compliance_issue(state: GraphState) -> dict[str, Any]:
    """Minimal report when no compliance issues emerge."""
    synthesis = state["synthesis"]
    report = Report(
        summary=synthesis["summary"],
        facts=state["facts"]["facts"],
        articles=state.get("candidate_articles", []),
        paragraphs=state.get("paragraphs", []),
        citations=synthesis["citations"],
        open_questions=[],
        unverified_citations=[
            c["chunk_id"] for c in synthesis["citations"] if not c["verified"]
        ],
    )
    return {"report": _apply_report_guards(report)}


def report_node(state: GraphState) -> dict[str, Any]:
    """Final report with open questions, unverified citations and disclaimer."""
    facts = state["facts"]["facts"]
    synthesis = state["synthesis"]

    open_questions: list[OpenQuestion] = []
    for f in facts:
        if f["status"] == "inferred":
            open_questions.append(
                OpenQuestion(
                    kind="inferred",
                    field=f["field"],
                    question=(
                        f"Ho assunto che {f['field']} = {f['value']} "
                        f"({f['rationale']}). Confermi?"
                    ),
                )
            )
        elif f["status"] == "unknown":
            open_questions.append(
                OpenQuestion(
                    kind="unknown",
                    field=f["field"],
                    question=(
                        f"Non ho informazioni su {f['field']}. "
                        "Puoi fornirle per una valutazione più precisa?"
                    ),
                )
            )
        elif f["status"] == "missing":
            open_questions.append(
                OpenQuestion(
                    kind="missing",
                    field=f["field"],
                    question=(
                        f"Il campo {f['field']} è necessario per decidere "
                        "l'applicabilità. Per favore, forniscilo."
                    ),
                )
            )

    unverified = [
        c["chunk_id"] for c in synthesis["citations"] if not c["verified"]
    ]

    report = Report(
        summary=synthesis["summary"],
        facts=facts,
        articles=state.get("candidate_articles", []),
        paragraphs=state.get("paragraphs", []),
        citations=synthesis["citations"],
        open_questions=open_questions,
        unverified_citations=unverified,
    )
    return {"report": _apply_report_guards(report)}


def error_node(state: GraphState) -> dict[str, Any]:
    """Minimal report in case of error, with disclaimer."""
    err = state.get("error") or "unknown error"
    logger.error("graph.error: %s", err)
    report = Report(
        summary=(
            "The analysis could not be completed due to an error "
            f"in the LLM: {err}. Please try again or rephrase your question."
        ),
        facts=state.get("facts", {"facts": [], "user_input": ""})["facts"],  # type: ignore[index]
        articles=state.get("candidate_articles", []),
        paragraphs=state.get("paragraphs", []),
        citations=[],
        open_questions=[],
        unverified_citations=[],
    )
    return {"report": _apply_report_guards(report)}


def route_after_any(state: GraphState) -> str:
    return "error" if state.get("error") else "continue"


def route_after_facts(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    return "ask_user" if state.get("needs_user_input") else "article_discovery"


def route_after_synthesis(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    return "no_compliance_issue" if state.get("is_compliant") else "report"