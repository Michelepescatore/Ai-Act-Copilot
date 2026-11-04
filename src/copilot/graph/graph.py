"""Build the LangGraph graph with multilingual support."""

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


# Node that detects the language and propagates the trace_id (thread_id) to the logs.
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


# Node that translates the input into English (or leaves it unchanged if already English).
def translate_input(state: GraphState, *, llm: LLMClient) -> dict:
    code = state.get("detected_language", "en")
    user_input = state.get("user_input", "")
    if code == "en":
        return {"user_input_en": user_input}
    translated = translate_to_english(user_input, llm)
    return {"user_input_en": translated}


# Node that translates the report back into the user's original language.
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


# Node that prepares the clarification questions BEFORE the interrupt:
#   1. nodes.ask_user reformulates the missing facts as natural questions (English)
#   2. translate_from_english translates them into the user's language
#   3. writes the result into state["pending_questions"]
#
# Keeping the LLM part here (and not in the node with the interrupt) is essential
# in LangGraph: on resume, the node with the interrupt is re-executed from the top,
# but the LLM calls are not repeated because pending_questions is already
# persisted by the checkpointer.
def prepare_questions(state: GraphState, *, llm: LLMClient) -> dict:
    result = nodes.ask_user(state, llm=llm)
    questions = result["questions"]

    code = state.get("detected_language", "en")
    if code != "en":
        questions = [translate_from_english(q, code, llm) for q in questions]

    return {"pending_questions": questions}


# ask_user node: interrupt only, idempotent on resume.
# On resume, LangGraph re-executes the node from the top, but interrupt() returns
# the resume value immediately without calling the LLM (pending_questions is
# already in the state, so there is nothing to recompute).
def _make_ask_user_node():
    def ask_user_node(state: GraphState) -> dict:
        questions = state.get("pending_questions", [])
        answer = interrupt({"questions": questions})
        return {"user_clarification_original": answer}

    return ask_user_node


# Build the graph with timing and trace_id on every node.
def build_graph(llm: LLMClient) -> StateGraph:
    graph = StateGraph(GraphState)

    # Every node is wrapped by timed_node to log ms and cost.
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


# Compile the graph with an in-memory checkpointer.
def compile_graph(llm: LLMClient | None = None):
    llm = llm or LLMClient()
    graph = build_graph(llm)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)