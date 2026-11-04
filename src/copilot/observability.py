"""Logging JSON strutturato e utility per timing e costo per nodo."""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, Callable

# Trace id condiviso tra tutti i log della stessa esecuzione.
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

# Flag per includere o meno il contenuto utente nei log. Default: no.
LOG_CONTENT = os.environ.get("LOG_CONTENT", "false").lower() in ("1", "true", "yes")

# Costo DeepSeek in USD per milione di token (cache miss).
COST_PER_1M_INPUT_USD = 0.14
COST_PER_1M_OUTPUT_USD = 0.28


def set_trace_id(trace_id: str) -> None:
    """Imposta il trace id nel contesto corrente."""
    trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """Restituisce il trace id corrente, o stringa vuota se non impostato."""
    return trace_id_var.get()


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Costo stimato in USD per la coppia (prompt, completion) di token."""
    return (
        prompt_tokens * COST_PER_1M_INPUT_USD / 1_000_000
        + completion_tokens * COST_PER_1M_OUTPUT_USD / 1_000_000
    )


class JsonFormatter(logging.Formatter):
    """Formatter che stampa una riga JSON per ogni record di log."""

    def format(self, record: logging.LogRecord) -> str:
        # Campi base sempre presenti.
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        # Campi extra passati con extra={"extra_fields": {...}}.
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "WARNING") -> None:
    """Configura il root logger per emettere JSON su stderr."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.WARNING))


def log_node(name: str, ms: float, prompt_tokens: int, completion_tokens: int) -> None:
    """Logga una riga JSON con timing e costo di un nodo del grafo."""
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
    """Avvolge un nodo per misurare tempo e costo.

    Se il nodo accetta un parametro `config` (come language_detect),
    il wrapper lo dichiara nella propria firma, così LangGraph lo passa.
    Altrimenti il wrapper resta `(*args, **kwargs)` e LangGraph non passa
    nulla di extra.
    """
    # Ispeziona la firma per capire se il nodo accetta `config`.
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
        # Se il nodo accetta `config`, il wrapper lo dichiara esplicitamente
        # nella firma, così LangGraph lo passa come secondo argomento.
        @functools.wraps(fn)
        def _run_with_config(state: Any, config: Any, **kwargs: Any) -> dict:
            return _run(state, config, **kwargs)

        return _run_with_config

    return _run