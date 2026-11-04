"""Costruzione del grafo LangGraph con supporto multilingua."""

from __future__ import annotations

from functools import partial

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from copilot.graph import nodes
from copilot.graph.language import (
    SUPPORTED_LANGUAGES,
    detect_language,
    translate_from_english,
    translate_to_english,
)
from copilot.graph.state import GraphState
from copilot.llm import LLMClient
from copilot.observability import set_trace_id, timed_node


# Nodo che rileva la lingua e propaga il trace_id (thread_id) ai log.
def language_detect(
    state: GraphState,
    config: RunnableConfig,
    *,
    llm: LLMClient,
) -> dict:
    tid = config["configurable"].get("thread_id", "") if config else ""
    if tid:
        set_trace_id(tid)

    user_input = state.get("user_input", "")
    code = detect_language(user_input, llm)
    if code not in SUPPORTED_LANGUAGES:
        code = "en"
    return {"detected_language": code, "trace_id": tid}


# Nodo che traduce l'input in inglese (o lo lascia invariato se già inglese).
def translate_input(state: GraphState, *, llm: LLMClient) -> dict:
    code = state.get("detected_language", "en")
    user_input = state.get("user_input", "")
    if code == "en":
        return {"user_input_en": user_input}
    translated = translate_to_english(user_input, llm)
    return {"user_input_en": translated}


# Nodo che traduce il report nella lingua originale dell'utente.
def translate_output(state: GraphState, *, llm: LLMClient) -> dict:
    code = state.get("detected_language", "en")
    report = state.get("report")
    if not report or code == "en":
        return {}
    translated_summary = translate_from_english(report["summary"], code, llm)
    translated_questions = []
    for q in report["open_questions"]:
        tq = translate_from_english(q["question"], code, llm)
        translated_questions.append({**q, "question": tq})
    new_report = {
        **report,
        "summary": translated_summary,
        "open_questions": translated_questions,
    }
    return {"report": new_report, "final_answer": translated_summary}


# Nodo che prepara le domande di chiarificazione PRIMA dell'interrupt:
#   1. nodes.ask_user riformula i fatti mancanti in domande naturali (inglese)
#   2. translate_from_english le traduce nella lingua utente
#   3. scrive il risultato in state["pending_questions"]
#
# Tenere qui la parte LLM (e non nel nodo con interrupt) è essenziale in
# LangGraph: al resume il nodo con interrupt viene rieseguito dall'inizio,
# ma le chiamate LLM non vengono rifatte perché pending_questions è già
# persistito dal checkpointer.
def prepare_questions(state: GraphState, *, llm: LLMClient) -> dict:
    result = nodes.ask_user(state, llm=llm)
    questions = result["questions"]

    code = state.get("detected_language", "en")
    if code != "en":
        questions = [translate_from_english(q, code, llm) for q in questions]

    return {"pending_questions": questions}


# Nodo ask_user: solo interrupt, idempotente al resume.
# Al resume LangGraph riesegue il nodo dall'inizio, ma interrupt() ritorna
# subito il valore del resume senza chiamare l'LLM (pending_questions è
# già nello state, quindi non c'è nulla da ricalcolare).
def _make_ask_user_node():
    def ask_user_node(state: GraphState) -> dict:
        questions = state.get("pending_questions", [])
        answer = interrupt({"questions": questions})
        return {"user_clarification_original": answer}

    return ask_user_node


# Costruisce il grafo con timing e trace_id su ogni nodo.
def build_graph(llm: LLMClient) -> StateGraph:
    graph = StateGraph(GraphState)

    # Ogni nodo viene avvolto da timed_node per loggare ms e costo.
    graph.add_node(
        "language_detect",
        timed_node("language_detect", partial(language_detect, llm=llm), llm),
    )
    graph.add_node(
        "translate_input",
        timed_node("translate_input", partial(translate_input, llm=llm), llm),
    )
    graph.add_node(
        "fact_extraction",
        timed_node("fact_extraction", partial(nodes.fact_extraction, llm=llm), llm),
    )
    graph.add_node(
        "prepare_questions",
        timed_node(
            "prepare_questions", partial(prepare_questions, llm=llm), llm
        ),
    )
    graph.add_node("ask_user", timed_node("ask_user", _make_ask_user_node(), None))
    graph.add_node(
        "article_discovery",
        timed_node("article_discovery", partial(nodes.article_discovery, llm=llm), llm),
    )
    graph.add_node(
        "paragraph_check",
        timed_node("paragraph_check", nodes.paragraph_check, None),
    )
    graph.add_node(
        "synthesis",
        timed_node("synthesis", partial(nodes.synthesis, llm=llm), llm),
    )
    graph.add_node(
        "no_compliance_issue",
        timed_node("no_compliance_issue", nodes.no_compliance_issue, None),
    )
    graph.add_node("report", timed_node("report", nodes.report_node, None))
    graph.add_node(
        "translate_output",
        timed_node("translate_output", partial(translate_output, llm=llm), llm),
    )
    graph.add_node("error", timed_node("error", nodes.error_node, None))

    graph.add_edge(START, "language_detect")
    graph.add_edge("language_detect", "translate_input")
    graph.add_edge("translate_input", "fact_extraction")

    graph.add_conditional_edges(
        "fact_extraction",
        nodes.route_after_facts,
        {
            "error": "error",
            "ask_user": "prepare_questions",
            "article_discovery": "article_discovery",
        },
    )
    graph.add_edge("prepare_questions", "ask_user")
    graph.add_edge("ask_user", "fact_extraction")

    graph.add_conditional_edges(
        "article_discovery",
        nodes.route_after_any,
        {"error": "error", "continue": "paragraph_check"},
    )
    graph.add_conditional_edges(
        "paragraph_check",
        nodes.route_after_any,
        {"error": "error", "continue": "synthesis"},
    )
    graph.add_conditional_edges(
        "synthesis",
        nodes.route_after_synthesis,
        {
            "error": "error",
            "no_compliance_issue": "no_compliance_issue",
            "report": "report",
        },
    )

    graph.add_edge("no_compliance_issue", "translate_output")
    graph.add_edge("report", "translate_output")
    graph.add_edge("translate_output", END)
    graph.add_edge("error", END)

    return graph


# Compila il grafo con un checkpointer in memoria.
def compile_graph(llm: LLMClient | None = None):
    llm = llm or LLMClient()
    graph = build_graph(llm)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)