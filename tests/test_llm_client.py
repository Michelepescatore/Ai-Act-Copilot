"""Tests for copilot.llm.LLMClient (OpenAI-based, json_object mode).

No network, no real `openai` import: the OpenAI client is replaced with a
MagicMock that mimics `client.chat.completions.create(...)`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from copilot.llm import LLMClient, LLMError


class _Answer(BaseModel):
    text: str
    n: int


def _mock_client_with(content: str) -> MagicMock:
    """Mimic `openai.OpenAI().chat.completions.create` response shape."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    # No `usage` by default; individual tests can override it.
    response.usage = None
    mock = MagicMock()
    mock.chat.completions.create.return_value = response
    return mock


def test_structured_complete_happy_path():
    client = _mock_client_with(json.dumps({"text": "ok", "n": 3}))
    llm = LLMClient(client=client)

    result = llm.structured_complete("Say ok", _Answer, system_prompt="Be brief.")

    assert isinstance(result, _Answer)
    assert result.text == "ok"
    assert result.n == 3

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0.0
    assert kwargs["model"] == llm.model

    # The Pydantic schema must be embedded in the system prompt.
    # We check structural keys, not the class title, so the test does not
    # break when the test model is renamed.
    system_content = kwargs["messages"][0]["content"]
    assert "JSON schema" in system_content
    assert '"properties"' in system_content
    assert '"text"' in system_content
    assert '"n"' in system_content
    assert '"required"' in system_content


def test_invalid_json_raises_llm_error():
    client = _mock_client_with("not json")
    llm = LLMClient(client=client)
    with pytest.raises(LLMError, match="invalid JSON"):
        llm.structured_complete("x", _Answer)


def test_schema_mismatch_raises_llm_error():
    client = _mock_client_with(json.dumps({"text": "ok"}))  # missing `n`
    llm = LLMClient(client=client)
    with pytest.raises(LLMError, match="does not match"):
        llm.structured_complete("x", _Answer)


def test_transport_error_raises_llm_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    llm = LLMClient(client=client)
    with pytest.raises(LLMError, match="LLM call failed"):
        llm.structured_complete("x", _Answer)


def test_empty_content_raises_llm_error():
    client = _mock_client_with("")
    llm = LLMClient(client=client)
    with pytest.raises(LLMError, match="empty content"):
        llm.structured_complete("x", _Answer)


def test_no_choices_raises_llm_error():
    client = MagicMock()
    response = MagicMock()
    response.choices = []
    response.usage = None
    client.chat.completions.create.return_value = response
    llm = LLMClient(client=client)
    with pytest.raises(LLMError, match="no choices"):
        llm.structured_complete("x", _Answer)


def test_token_counters_are_updated():
    client = _mock_client_with(json.dumps({"text": "ok", "n": 1}))
    response = client.chat.completions.create.return_value
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    response.usage = usage

    llm = LLMClient(client=client)
    llm.structured_complete("x", _Answer)
    llm.structured_complete("y", _Answer)

    assert llm.total_prompt_tokens == 20
    assert llm.total_completion_tokens == 10


def test_missing_api_key_raises_in_clean_process(tmp_path):
    """LLMClient() without a key must fail.

    Cannot test in-process because copilot/__init__.py calls load_dotenv(),
    so os.environ may already contain DEEPSEEK_API_KEY from .env. A subprocess
    with an empty cwd (no .env) and an empty environment is the only honest
    way to test this path.
    """
    code = (
        "from copilot.llm import LLMClient, LLMError\n"
        "try:\n"
        "    LLMClient()\n"
        "except LLMError as exc:\n"
        "    print('OK' if 'DEEPSEEK_API_KEY' in str(exc) else 'WRONG:' + str(exc))\n"
        "else:\n"
        "    print('NO_ERROR')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={},
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "OK", result.stdout + result.stderr