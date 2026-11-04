"""Structured JSON logging and utilities for per-node timing and cost.

We do not use structlog: standard logging + a custom formatter is enough.
The LOG_CONTENT flag (env, default "false") excludes user text from logs:
useful in public demos.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, Callable

# Trace id shared by all logs of the same execution.
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

# Flag to include or exclude user content in logs. Default: no.
LOG_CONTENT = os.environ.get("LOG_CONTENT", "false").lower() in ("1", "true", "yes")

# DeepSeek cost in USD per million tokens (cache miss).
COST_PER_1M_INPUT_USD = 0.14
COST_PER_1M_OUTPUT_USD = 0.28


def set_trace_id(trace_id: str) -> None:
    """Set the trace id in the current context."""
    trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """Return the current trace id, or an empty string if not set."""
    return trace_id_var.get()


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated cost in USD for the (prompt, completion) token pair."""
    return (
        prompt_tokens * COST_PER_1M_INPUT_USD / 1_000_000
        + completion_tokens * COST_PER_1M_OUTPUT_USD / 1_000_000
    )


class JsonFormatter(logging.Formatter):
    """Formatter that prints one JSON line for each log record."""

    def format(self, record: logging.LogRecord) -> str:
        # Base fields always present.
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        # Extra fields passed with extra={"extra_fields": {...}}.
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "WARNING") -> None:
    """Configure the root logger to emit JSON on stderr."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.WARNING))


def log_node(name: str, ms: float, prompt_tokens: int, completion_tokens: int) -> None:
    """Log a JSON line with the timing and cost of a graph node."""
    cost = estimate_cost(prompt_tokens, completion_tokens)
    logging.getLogger("copilot.nodes").info(
        "node_executed",
        extra={
            "extra_fields": {
                "node": name,
                "ms": round(ms, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(cost, 6),
            }
        },
    )


def timed_node(
    name: str,
    fn: Callable[..., dict],
    llm: Any | None = None,
) -> Callable[..., dict]:
    """Wrap a node to measure time and cost.

    If the node accepts a `config` parameter (like language_detect),
    the wrapper declares it in its own signature, so LangGraph passes it.
    Otherwise the wrapper stays `(*args, **kwargs)` and LangGraph does not
    pass anything extra.
    """
    # Inspect the signature to understand whether the node accepts `config`.
    try:
        sig = inspect.signature(fn)
        accepts_config = "config" in sig.parameters
    except (ValueError, TypeError):
        accepts_config = False

    @functools.wraps(fn)
    def _run(*args: Any, **kwargs: Any) -> dict:
        start = time.perf_counter()
        p0 = getattr(llm, "total_prompt_tokens", 0) if llm else 0
        c0 = getattr(llm, "total_completion_tokens", 0) if llm else 0
        try:
            return fn(*args, **kwargs)
        finally:
            ms = (time.perf_counter() - start) * 1000
            p1 = getattr(llm, "total_prompt_tokens", 0) if llm else 0
            c1 = getattr(llm, "total_completion_tokens", 0) if llm else 0
            log_node(name, ms, p1 - p0, c1 - c0)

    if accepts_config:
        # If the node accepts `config`, the wrapper declares it explicitly
        # in the signature, so LangGraph passes it as the second argument.
        @functools.wraps(fn)
        def _run_with_config(state: Any, config: Any, **kwargs: Any) -> dict:
            return _run(state, config, **kwargs)

        return _run_with_config

    return _run