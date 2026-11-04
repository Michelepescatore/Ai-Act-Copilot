"""CLI orchestrator for the AI Act Compliance Copilot."""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import Any

from langgraph.types import Command

from copilot.graph.graph import compile_graph
from copilot.graph.state import Report
from copilot.llm import LLMClient, LLMError
from copilot.observability import setup_logging

_SEP = "─" * 72


def _print_header(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def _print_report(report: Report) -> None:
    """Print the report with sources, unverified citations, questions, and articles."""
    _print_header("RESULT")
    print(report["summary"].strip())

    citations = report.get("citations", [])
    if citations:
        print("\nSources:")
        for c in citations:
            mark = "✓" if c.get("verified") else "✗"
            print(f"  {mark} [{c['chunk_id']}] {c['claim']}")

    unverified = report.get("unverified_citations", [])
    if unverified:
        print("\n⚠ Unverified citations:")
        for chunk_id in unverified:
            print(f"  - {chunk_id}")

    questions = report.get("open_questions", [])
    if questions:
        print("\nOpen questions:")
        for q in questions:
            print(f"  [{q['kind']}] {q['question']}")

    articles = report.get("articles", [])
    if articles:
        print("\nConsidered articles:")
        for a in articles:
            print(f"  - {a['id']}: {a['title']}")


def _print_clarification(questions: list[str]) -> None:
    _print_header("CLARIFICATION REQUIRED")
    for i, q in enumerate(questions, start=1):
        print(f"  {i}. {q}")
    print()


def _print_not_ai_act() -> None:
    _print_header("IRRELEVANT INPUT")
    print(
        "The provided text does not seem to describe an artificial intelligence "
        "system nor a case attributable to the AI Act.\n\n"
        "Try to describe the operator role, the type of system, and the sector."
    )


def _print_error(err: str) -> None:
    _print_header("ERROR")
    print(f"Processing interrupted: {err}\n")


def _handle_llm_error(exc: LLMError, result: dict | None) -> int:
    """Handle the LLM error. If it is budget_exceeded, show the partial report."""
    err_str = str(exc)
    if "budget_exceeded" in err_str:
        print(f"\n⚠ LLM budget exhausted: {err_str}", file=sys.stderr)
        report = (result or {}).get("report")
        if report:
            print("\nPartial report (may be incomplete):")
            _print_report(report)
        else:
            print("\nNo partial report available.")
        return 0
    _print_error(f"LLM unavailable: {err_str}")
    return 1


def _is_not_ai_act(result: dict[str, Any]) -> bool:
    """Heuristic: no facts extracted => probably out of scope."""
    facts = result.get("facts") or {}
    return len(facts.get("facts", [])) == 0


def _read_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _run_once(*, app: Any, config: dict[str, Any], user_input: str) -> dict[str, Any]:
    return app.invoke({"user_input": user_input}, config=config)


def _resume(*, app: Any, config: dict[str, Any], answer: str) -> dict[str, Any]:
    return app.invoke(Command(resume=answer), config=config)


def _handle_result(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify the graph result into clarification / error / report."""
    if "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        payload = getattr(interrupts[0], "value", None) or {}
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not questions:
            questions = [str(payload)]
        return "clarification", {"questions": questions}

    if result.get("error"):
        return "error", {"error": result["error"]}

    if _is_not_ai_act(result):
        return "report", {"_not_ai_act": True}

    report = result.get("report")
    if not report:
        return "error", {"error": "The graph did not produce a report."}

    return "report", report


def run_session(
    *,
    case: str | None,
    thread_id: str | None,
    max_clarifications: int = 5,
    llm_client: LLMClient | None = None,
) -> int:
    """Run an interactive session. Returns an exit code."""
    llm = llm_client or LLMClient()
    app = compile_graph(llm=llm)

    tid = thread_id or f"cli:{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": tid}}

    # PII warning: the text is sent to an external LLM.
    print(
        "⚠ Do not enter real personal data: the text is translated and "
        "sent to DeepSeek.",
        file=sys.stderr,
    )
    print(f"Session: {tid}")

    if case:
        user_input = case
        print(f"\nCase: {user_input}")
    else:
        print("\nDescribe the case (one line, press Enter to confirm):")
        user_input = _read_line("> ")
        if not user_input:
            print("No input provided. Exiting.")
            return 1

    try:
        result = _run_once(app=app, config=config, user_input=user_input)
    except LLMError as exc:
        return _handle_llm_error(exc, None)

    clarifications = 0
    while True:
        kind, payload = _handle_result(result)

        if kind == "clarification":
            clarifications += 1
            if clarifications > max_clarifications:
                _print_error(
                    f"Reached the limit of {max_clarifications} clarifications."
                )
                return 1
            _print_clarification(payload["questions"])
            answer = _read_line("Answer: ")
            if not answer:
                _print_error("No answer provided. Exiting.")
                return 1
            try:
                result = _resume(app=app, config=config, answer=answer)
            except LLMError as exc:
                return _handle_llm_error(exc, result)
            continue

        if kind == "error":
            _print_error(payload["error"])
            return 1

        if payload.get("_not_ai_act"):
            _print_not_ai_act()
            return 2

        _print_report(payload)  # type: ignore[arg-type]
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="copilot.cli",
        description="Assess a case against the AI Act (Regulation (EU) 2024/1689).",
    )
    parser.add_argument("--case", default=None, help="Description of the case.")
    parser.add_argument("--thread", default=None, help="Thread ID for the session.")
    parser.add_argument(
        "--max-clarifications", type=int, default=5,
        help="Maximum number of clarification rounds (default 5).",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="JSON logging level on stderr (default WARNING).",
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    try:
        return run_session(
            case=args.case,
            thread_id=args.thread,
            max_clarifications=args.max_clarifications,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by the user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())