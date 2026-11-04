"""MCP server (mcp 1.x) exposing the AI Act corpus and the compliance graph.

Tools: search_regulation, get_article, get_annex_point,
get_applicability_date, assess_ai_system.

Auth: none. Rate limiting is applied to mitigate abuse on public tunnels.

Allineato al codice reale:
- `QdrantStore(url=...)` non accetta `collection` nel costruttore.
- `Retriever.search(query, *, collection=, method=, limit=)` richiede
  `collection` e usa `method`/`limit` (non `mode`/`k`).
- Il corpus è in due collezioni: `articles_v1` (profilo semantico) e
  `paragraphs_v1` (testo verbatim).
- Il grafo si invoca via `copilot.cli.run_session`? No: qui serve un
  runner programmatico. Usiamo direttamente `compile_graph` + `invoke`,
  replicando la logica di `cli.py` ma senza input/output interattivi.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "ai-act-copilot",
    instructions=(
        "Tools to query the consolidated text of Regulation (EU) 2024/1689 "
        "(AI Act) as amended by Regulation (EU) 2026/1744, and to assess an "
        "AI system against it. This is not legal advice.\n\n"
        "IMPORTANT: when the user asks whether a described AI system is "
        "legal/compliant under the AI Act, ALWAYS call `assess_ai_system` "
        "first, before answering from prior knowledge. The tool returns "
        "either `{status: 'completed', report: {...}}` or "
        "`{status: 'clarification_needed', questions: [...], thread_id: '...'}`.\n\n"
        "If the tool returns `clarification_needed`, show the questions to "
        "the user and wait for their answer. Then call `assess_ai_system` "
        "again with the SAME `thread_id` and the user's reply as `answer`. "
        "Repeat until `status: 'completed'`."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


# ── Rate limiting (in-memory, per IP) ────────────────────────────────────────
_RATE_LIMIT = int(os.environ.get("MCP_RATE_LIMIT", "30"))
_RATE_WINDOW = int(os.environ.get("MCP_RATE_WINDOW", "60"))
_requests: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if _RATE_LIMIT <= 0:
            return await call_next(request)
        ip = (
            request.headers.get("cf-connecting-ip")
            or (request.client.host if request.client else "unknown")
        )
        now = time.time()
        window = _requests[ip]
        window[:] = [t for t in window if now - t < _RATE_WINDOW]
        if len(window) >= _RATE_LIMIT:
            retry_after = int(_RATE_WINDOW - (now - window[0])) + 1
            return JSONResponse(
                {"error": "rate limit exceeded", "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)
        return await call_next(request)


# ── Retriever singleton ─────────────────────────────────────────────────────
_retriever: Any = None


def _get_retriever() -> Any:
    global _retriever
    if _retriever is None:
        from copilot.retrieval.embedders import Embedders
        from copilot.retrieval.search import Retriever
        from copilot.retrieval.store import QdrantStore

        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        _retriever = Retriever(
            embedders=Embedders(),
            store=QdrantStore(url=url),
        )
    return _retriever


# ── Graph singleton (per riusare il checkpointer tra le chiamate) ───────────
_graph_app: Any = None


def _get_graph_app() -> Any:
    global _graph_app
    if _graph_app is None:
        from copilot.graph.graph import compile_graph
        from copilot.llm import LLMClient

        _graph_app = compile_graph(llm=LLMClient())
    return _graph_app


# Mappa doc_type → collezione Qdrant.
_DOC_TYPE_TO_COLLECTION = {
    "article": "articles_v1",
    "articles": "articles_v1",
    "annex": "articles_v1",
    "paragraph": "paragraphs_v1",
    "paragraphs": "paragraphs_v1",
}


# ── Tool: search_regulation ─────────────────────────────────────────────────
@mcp.tool()
def search_regulation(
    query: str,
    k: int = 5,
    mode: str = "hybrid_rerank",
    doc_type: str | None = None,
) -> list[dict[str, Any]]:
    """Search the consolidated AI Act text.

    Args:
        query: natural-language query.
        k: number of results (1-20).
        mode: one of "dense", "sparse", "hybrid", "hybrid_rerank".
        doc_type: optional filter. "paragraph" → paragraphs_v1 (verbatim),
            "article"/"annex" → articles_v1 (semantic profile).
    """
    k = max(1, min(int(k), 20))
    collection = _DOC_TYPE_TO_COLLECTION.get(
        (doc_type or "paragraph").lower(), "paragraphs_v1"
    )
    results = _get_retriever().search(
        query,
        collection=collection,
        method=mode,  # type: ignore[arg-type]
        limit=k,
    )
    return [
        {
            "id": r["id"],
            "score": r["score"],
            "doc_type": r["payload"].get("doc_type"),
            "article": r["payload"].get("article"),
            "annex": r["payload"].get("annex"),
            "paragraph": r["payload"].get("paragraph"),
            "chunk_id": r["payload"].get("chunk_id"),
            "title": r["payload"].get("title")
            or r["payload"].get("article_title"),
            "text": r["payload"].get("text")
            or r["payload"].get("verbatim_excerpt"),
            "url": r["payload"].get("url"),
        }
        for r in results
    ]


# ── Tool: get_article ───────────────────────────────────────────────────────
@mcp.tool()
def get_article(article: str, paragraph: str | None = None) -> dict[str, Any] | None:
    """Return the exact text of an article of the AI Act.

    `article` accepts "6", "art_6", "Art. 6". `paragraph` is optional.
    """
    # Normalizza il numero articolo.
    article_num = article.lower().replace("art.", "").replace("art_", "").strip()
    article_num = article_num.lstrip("0") or article_num

    query = f"article {article_num}"
    if paragraph:
        query += f" paragraph {paragraph}"

    hits = _get_retriever().search(
        query,
        collection="paragraphs_v1",
        method="hybrid_rerank",
        limit=20,
    )

    def _match(h: dict) -> bool:
        p = h["payload"]
        chunk = p.get("chunk_id", "")
        if str(p.get("article", "")) == article_num:
            return True
        return chunk.startswith(f"art_{article_num}.") or chunk == f"art_{article_num}"

    candidates = [h for h in hits if _match(h)]
    if paragraph:
        candidates = [
            h for h in candidates
            if str(h["payload"].get("paragraph", "")) == str(paragraph)
            or h["payload"].get("chunk_id", "").endswith(f".p{paragraph}")
        ]
    if not candidates:
        return None
    top = candidates[0]["payload"]
    return {
        "chunk_id": top.get("chunk_id"),
        "article": top.get("article"),
        "paragraph": top.get("paragraph"),
        "title": top.get("article_title"),
        "text": top.get("text"),
        "url": top.get("url"),
    }


# ── Tool: get_annex_point ───────────────────────────────────────────────────
@mcp.tool()
def get_annex_point(annex: str, point: str | None = None) -> dict[str, Any] | None:
    """Return the exact text of a point of an annex of the AI Act."""
    annex_num = (
        annex.lower()
        .replace("annex", "")
        .replace("anx_", "")
        .replace("anx", "")
        .strip()
    )

    query = f"annex {annex_num}"
    if point:
        query += f" point {point}"

    hits = _get_retriever().search(
        query,
        collection="articles_v1",
        method="hybrid_rerank",
        limit=20,
    )

    def _match(h: dict) -> bool:
        p = h["payload"]
        return str(p.get("annex", "")) == annex_num or p.get("id", "").startswith(
            f"anx_{annex_num}"
        )

    candidates = [h for h in hits if _match(h)]
    if point:
        candidates = [
            h for h in candidates
            if h["payload"].get("id", "").endswith(f".{point}")
            or str(h["payload"].get("point", "")) == str(point)
        ]
    if not candidates:
        return None
    top = candidates[0]["payload"]
    return {
        "id": top.get("id"),
        "annex": top.get("annex"),
        "title": top.get("title"),
        "summary": top.get("summary"),
        "verbatim_excerpt": top.get("verbatim_excerpt"),
        "url": top.get("url"),
    }


# ── Tool: get_applicability_date ────────────────────────────────────────────
@mcp.tool()
def get_applicability_date(article: str, risk_tier: str) -> str:
    """Return the application date of an article for a given risk tier."""
    from copilot.schemas import RiskTier
    from copilot.timeline import applies_from

    tier = RiskTier(risk_tier)
    result = applies_from(article, tier)
    return result.date.isoformat()


# ── Tool: assess_ai_system ──────────────────────────────────────────────────
def _extract_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    """Estrae il payload di `__interrupt__` (formato nuovo: questions list)."""
    if "__interrupt__" not in result:
        return None
    payload = getattr(result["__interrupt__"][0], "value", None)
    if isinstance(payload, dict):
        questions = payload.get("questions") or []
        if not questions and payload.get("question"):
            questions = [payload["question"]]
    else:
        questions = [str(payload)] if payload else []
    return {"questions": questions}


def _serialize_report(report: Any) -> dict[str, Any]:
    """Normalizza il report in dict JSON-serializzabile."""
    if report is None:
        return {}
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")
    return dict(report)


@mcp.tool()
def assess_ai_system(
    description: str,
    thread_id: str | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    """Assess an AI system against the AI Act and return a structured report.

    Two modes:

    - New assessment: pass `description` (and optionally `thread_id`).
      If the graph needs clarification, returns
      {"status": "clarification_needed", "questions": [...], "thread_id": "..."}.
      The caller must call again with the same `thread_id` and `answer`.

    - Resume: pass `thread_id` and `answer` (the user's reply).
      Returns {"status": "completed", "report": {...}} or another
      clarification_needed if new questions arise.
    """
    from langgraph.types import Command

    app = _get_graph_app()

    if thread_id and answer is not None:
        # Resume del grafo.
        config = {"configurable": {"thread_id": thread_id}}
        result = app.invoke(Command(resume=answer), config=config)
        interrupt_payload = _extract_interrupt(result)
        if interrupt_payload is not None:
            return {
                "status": "clarification_needed",
                "questions": interrupt_payload["questions"],
                "thread_id": thread_id,
            }
        if result.get("error"):
            return {"status": "error", "error": result["error"], "thread_id": thread_id}
        return {
            "status": "completed",
            "report": _serialize_report(result.get("report")),
        }

    # Nuova valutazione.
    tid = thread_id or f"mcp:{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": tid}}
    result = app.invoke({"user_input": description}, config=config)
    interrupt_payload = _extract_interrupt(result)
    if interrupt_payload is not None:
        return {
            "status": "clarification_needed",
            "questions": interrupt_payload["questions"],
            "thread_id": tid,
        }
    if result.get("error"):
        return {"status": "error", "error": result["error"], "thread_id": tid}
    return {
        "status": "completed",
        "report": _serialize_report(result.get("report")),
    }