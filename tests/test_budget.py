"""Tests for the LLM budget.

The budget is expressed in USD and is incremented on every call based on
the tokens consumed (usage of the response). The check happens before every
call: if the accumulated cost has already exceeded the limit, the call
is blocked with LLMError("budget_exceeded: ...").
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from copilot.llm import LLMClient, LLMError


class _Schema(BaseModel):
    x: int


def _fake_client_with_usage(
    content: str = '{"x": 1}',
    prompt: int = 10_000,
    completion: int = 10_000,
) -> MagicMock:
    """Fake OpenAI client that consumes tokens on every call."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    response.usage = usage

    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_budget_zero_blocks_first_call():
    """With max_cost_usd=0.0 the check fires immediately: no call is made."""
    client = _fake_client_with_usage()
    llm = LLMClient(client=client, max_cost_usd=0.0)

    with pytest.raises(LLMError, match="budget_exceeded"):
        llm.structured_complete("x", _Schema)

    # The client was not called because the budget was already exhausted.
    client.chat.completions.create.assert_not_called()


def test_budget_exceeded_after_first_call():
    """With a small budget, the first call passes, the second is blocked."""
    client = _fake_client_with_usage()
    llm = LLMClient(client=client, max_cost_usd=0.001)

    # First call: cost_usd=0.0 < 0.001, proceeds.
    llm.structured_complete("x", _Schema)
    # After: cost_usd ≈ 0.0042 > 0.001.
    assert llm.cost_usd > 0.001

    # Second call: 0.0042 >= 0.001, blocked.
    with pytest.raises(LLMError, match="budget_exceeded"):
        llm.structured_complete("y", _Schema)


def test_budget_allows_calls_under_limit():
    """With a large budget, calls pass and the cost is accumulated."""
    client = _fake_client_with_usage()
    llm = LLMClient(client=client, max_cost_usd=1.0)

    llm.structured_complete("x", _Schema)
    assert 0 < llm.cost_usd < 1.0
    assert llm.total_prompt_tokens == 10_000
    assert llm.total_completion_tokens == 10_000