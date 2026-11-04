"""Unit test di `copilot.graph.language`."""

from __future__ import annotations

from copilot.graph.language import (
    SUPPORTED_LANGUAGES,
    detect_language,
    translate_from_english,
    translate_to_english,
)
from copilot.llm import LLMError


def test_detect_italian(fake_llm):
    llm = fake_llm([{"code": "it"}])
    assert detect_language("Ciao, come stai?", llm) == "it"


def test_detect_normalizes_region(fake_llm):
    llm = fake_llm([{"code": "IT-it"}])
    assert detect_language("Ciao", llm) == "it"


def test_detect_normalizes_underscore(fake_llm):
    llm = fake_llm([{"code": "en_US"}])
    assert detect_language("Hello", llm) == "en"


def test_detect_llm_error_falls_back_to_english(fake_llm):
    llm = fake_llm([LLMError("boom")])
    assert detect_language("Ciao", llm) == "en"


def test_translate_to_english_ok(fake_llm):
    llm = fake_llm([{"text": "Hello, how are you?"}])
    assert translate_to_english("Ciao, come stai?", llm) == "Hello, how are you?"


def test_translate_to_english_llm_error_returns_original(fake_llm):
    llm = fake_llm([LLMError("boom")])
    assert translate_to_english("Ciao", llm) == "Ciao"


def test_translate_from_english_english_target_is_noop(fake_llm):
    llm = fake_llm([])
    assert translate_from_english("hello", "en", llm) == "hello"


def test_translate_from_english_unsupported_target_is_noop(fake_llm):
    llm = fake_llm([])
    assert translate_from_english("hello", "xx", llm) == "hello"


def test_translate_from_english_supported_target_calls_llm(fake_llm):
    assert "it" in SUPPORTED_LANGUAGES
    llm = fake_llm([{"text": "ciao"}])
    assert translate_from_english("hello", "it", llm) == "ciao"
    assert len(llm.calls) == 1
    assert "italiano" in llm.calls[0]["system_prompt"]


def test_translate_from_english_llm_error_returns_english(fake_llm):
    llm = fake_llm([LLMError("boom")])
    assert translate_from_english("hello", "it", llm) == "hello"