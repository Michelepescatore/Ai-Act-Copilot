"""Nodi del grafo LangGraph."""

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

# Metodo di retrieval usato dal grafo.
RETRIEVAL_METHOD = "hybrid_rerank"

# Lunghezza massima dell'input inglese. Oltre, rifiutiamo con un errore gestito.
MAX_INPUT_CHARS = 4000

# Valori ammessi per i campi con vocabolario chiuso. Servono per suggerire
# le opzioni nella domanda di chiarificazione.
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

# Pattern di prompt injection. Non blocchiamo, ma segnaliamo.
_INJECTION_PATTERNS = [
    re.compile(r"\bignore (all |any )?(previous |prior )?instructions\b", re.I),
    re.compile(r"\bignore (the )?(system|prompt|rules)\b", re.I),
    re.compile(r"\bdisregard (all |any )?(previous |prior )?instructions\b", re.I),
    re.compile(r"\bforget (all |any )?(previous |prior )?instructions\b", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
]

# Frasi vietate nell'output. Se il modello le produce, vengono sostituite.
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


# Schema per la riformulazione delle domande di chiarificazione.
class _QuestionReformulationLLM(BaseModel):
    questions: list[str]


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    """Singleton del retriever. I test lo patchano via cache_clear."""
    from copilot.retrieval.embedders import Embedders

    return Retriever(
        embedders=Embedders(),
        store=QdrantStore(url="http://localhost:6333"),
    )


def _query_from_facts(facts: CaseFacts) -> str:
    """Costruisce la query di retrieval a partire dai fatti noti o inferiti."""
    parts = [
        f"{f['field']}={f['value']}"
        for f in facts["facts"]
        if f["status"] in ("known", "inferred") and f["value"]
    ]
    return " ".join(parts) if parts else facts["user_input"]


def _has_suspicious_pattern(text: str) -> bool:
    """True se il testo contiene un pattern di prompt injection noto."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def _neutralize_forbidden(text: str) -> str:
    """Sostituisce le frasi vietate con una formula neutra."""
    result = text
    for pat in FORBIDDEN_PATTERNS:
        result, n = pat.subn(_NEUTRAL_FALLBACK, result)
        if n:
            logger.warning("forbidden phrase neutralized: %s", pat.pattern)
    return result


def _apply_report_guards(report: dict) -> dict:
    """Neutralizza frasi vietate, aggiunge disclaimer e warning su citazioni."""
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
    """Estrae i fatti dal caso (in inglese, dopo la traduzione)."""
    from copilot.graph.language import translate_to_english

    user_input_en = state.get("user_input_en") or state.get("user_input", "")
    clarification_original = state.get("user_clarification_original")
    code = state.get("detected_language", "en")

    # Rifiuta input troppo lunghi prima di chiamare il modello.
    if len(user_input_en) > MAX_INPUT_CHARS:
        return {"error": f"input_too_long: {len(user_input_en)} > {MAX_INPUT_CHARS}"}

    # Rileva eventuale prompt injection. Non blocca: prosegue e segnala.
    suspicious = _has_suspicious_pattern(user_input_en)
    if suspicious:
        logger.warning("suspicious input detected (prompt injection pattern)")

    # Traduce il chiarimento dell'utente (se presente) in inglese.
    clarification_en = ""
    if clarification_original:
        if code != "en":
            clarification_en = translate_to_english(clarification_original, llm)
        else:
            clarification_en = clarification_original

    # Costruisce il prompt con input delimitato da tag.
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

    # Scarta i fatti "unknown" non obbligatori: il prompt dovrebbe già
    # ometterli, questa è una rete di sicurezza.
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
    """Riformula in linguaggio naturale le domande per l'utente.

    Prende i fatti obbligatori mancanti, costruisce un prompt con field,
    rationale e valori ammessi (se presenti), e chiede all'LLM di produrre
    una domanda leggibile da un utente non esperto. Se la chiamata fallisce,
    ricade su un template semplice.
    """
    facts = state["facts"]["facts"]
    blocking = [
        f for f in facts
        if f["required"] and f["status"] in ("unknown", "missing")
    ]
    if not blocking:
        return {"questions": []}

    # Costruisce la descrizione dei fatti mancanti per il prompt.
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
        # Fallback: template semplice, coerente con il numero di fatti.
        logger.warning("ask_user reformulation failed: %s", exc)
        questions = [
            f"Please provide a value for `{f['field']}`." for f in blocking
        ]

    return {"questions": questions}  # type: ignore[typeddict-unknown-key]


def article_discovery(state: GraphState, *, llm: LLMClient) -> dict[str, Any]:
    """Seleziona gli articoli candidati via retrieval + LLM."""
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
        f"Fatti del caso:\n{query}\n\n"
        f"Articoli candidati recuperati:\n{candidates_text}"
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
    """Recupera i paragrafi rilevanti in un'unica passata, poi filtra per articolo.

    Ottimizzazione: prima facevamo una chiamata di retrieval per ogni articolo
    candidato. Ora ne facciamo una sola con limit più alto, poi filtriamo per
    articolo in memoria.
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

    # Indicizza gli hit per numero di articolo, per filtro O(1).
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

        # Fallback: se l'articolo non ha hit diretti, prendi i primi 3 globali.
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
    """Produce la sintesi e verifica le citazioni contro il contesto."""
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
        f"Fatti del caso:\n{facts_text}\n\n"
        f"Articoli candidati:\n{articles_text}\n\n"
        f"Paragrafi rilevanti:\n{paragraphs_text}\n\n"
        f"Domanda originale: {facts['user_input']}"
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
    """Report minimale quando non emergono problemi di conformità."""
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
    """Report finale con domande aperte, citazioni non verificate e disclaimer."""
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
    """Report minimale in caso di errore, con disclaimer."""
    err = state.get("error") or "errore sconosciuto"
    logger.error("graph.error: %s", err)
    report = Report(
        summary=(
            "Non è stato possibile completare l'analisi a causa di un errore "
            f"nel LLM: {err}. Riprova o riformula la domanda."
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