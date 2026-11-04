"""Client LLM OpenAI-compatibile puntato a DeepSeek.

DeepSeek supporta `response_format={"type": "json_object"}` ma non
`json_schema`. Per questo descriviamo lo schema Pydantic nel system prompt
e validiamo la risposta contro il modello.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from copilot.observability import estimate_cost

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-chat"

SchemaType = TypeVar("SchemaType", bound=BaseModel)

# Default di budget per sessione. Modificabile via costruttore.
_DEFAULT_MAX_COST_USD = 0.50


class LLMError(RuntimeError):
    """Eccezione unica per qualsiasi problema LLM."""


def _is_retryable(exc: Exception) -> bool:
    # Retry solo su rate limit e timeout. Nome classe invece di isinstance
    # per non importare openai nei test che iniettano un fake.
    name = type(exc).__name__
    return name in ("RateLimitError", "APITimeoutError", "Timeout")


class LLMClient:
    """Client DeepSeek. Una istanza per processo."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_cost_usd: float = _DEFAULT_MAX_COST_USD,
    ) -> None:
        # Contatori cumulativi (compatibilità con chiamanti esistenti).
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.cost_usd: float = 0.0
        self.max_cost_usd = max_cost_usd

        self.model = model or os.environ.get("DEEPSEEK_MODEL", _DEFAULT_MODEL)
        self.temperature = temperature

        # Seam di test: con client iniettato non tocchiamo openai né env.
        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "openai not installed; run `uv add openai`."
            ) from exc

        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise LLMError("DEEPSEEK_API_KEY not set and no api_key provided.")

        url = base_url or os.environ.get("DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL)
        self._client = OpenAI(api_key=key, base_url=url, timeout=timeout)

    @property
    def client(self) -> Any:
        return self._client

    def _call_api(self, full_system: str, user_prompt: str) -> Any:
        # 1 retry su rate limit / timeout, poi fail pulito.
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and _is_retryable(exc):
                    logger.warning("LLM retry: %s", exc)
                    time.sleep(2)
                    continue
                break
        raise LLMError(f"LLM call failed: {last_exc}") from last_exc

    def structured_complete(
        self,
        user_prompt: str,
        schema: type[SchemaType],
        *,
        system_prompt: str = "",
    ) -> SchemaType:
        # Budget check: fail prima di bruciare altri token.
        if self.cost_usd >= self.max_cost_usd:
            raise LLMError(
                f"budget_exceeded: ${self.cost_usd:.4f} >= "
                f"${self.max_cost_usd:.4f}"
            )

        # Serializza lo schema nel system prompt (DeepSeek non supporta
        # json_schema direttamente).
        try:
            schema_json = schema.model_json_schema()
        except Exception as exc:
            raise LLMError(
                f"model_json_schema() failed for {schema.__name__}: {exc}"
            ) from exc

        schema_instruction = (
            "Respond with a single JSON object that strictly matches the "
            "following JSON schema. Do not include any text outside the JSON.\n\n"
            f"JSON schema:\n{json.dumps(schema_json, indent=2)}"
        )
        full_system = (
            f"{system_prompt}\n\n{schema_instruction}".strip()
            if system_prompt
            else schema_instruction
        )

        response = self._call_api(full_system, user_prompt)

        # Aggiorna token cumulativi e costo sessione.
        usage = getattr(response, "usage", None)
        if usage is not None:
            p = getattr(usage, "prompt_tokens", 0) or 0
            c = getattr(usage, "completion_tokens", 0) or 0
            self.total_prompt_tokens += p
            self.total_completion_tokens += c
            self.cost_usd += estimate_cost(p, c)
            logger.info(
                "llm: schema=%s tokens(p=%d, c=%d) cost=$%.6f",
                schema.__name__, p, c, self.cost_usd,
            )

        # Empty content è un errore fatale.
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("LLM returned no choices.")
        content = getattr(choices[0].message, "content", "") or ""
        if not content.strip():
            raise LLMError("LLM returned empty content.")

        # Parse JSON.
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}") from exc

        # Valida contro il modello Pydantic.
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise LLMError(
                f"LLM output does not match {schema.__name__}: {exc}"
            ) from exc