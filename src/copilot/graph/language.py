"""Language detection and translation.

Detects the language of the user's input via LLM, then translates it into
English before the pipeline and back into the original language before
the report.

If the language is not in `languages.json` or is already English, the
translation is skipped and the flow uses the original input.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from copilot.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

LANGUAGE_FILE = Path(__file__).parent / "languages.json"


def _load_supported_languages() -> dict[str, str]:
    """Load the code → name mapping from `languages.json`."""
    with LANGUAGE_FILE.open(encoding="utf-8") as f:
        return json.load(f)


# Module-level cache of the mapping (the file does not change at runtime)
SUPPORTED_LANGUAGES = _load_supported_languages()


class _LanguageDetection(BaseModel):
    """Schema of the LLM response for language detection."""
    code: str  # ISO 639-1, e.g. "it", "en", "ja"


_LANGUAGE_DETECTION_SYSTEM = """\
You are a language detector. Given a text string, return the ISO 639-1 code
of the main language (two letters, e.g. "it", "en", "ja", "ar").
If the text is ambiguous, choose the dominant language.
"""


_TRANSLATE_TO_EN_SYSTEM = """\
You are a professional translator. Translate the user's text into English.
Faithfully preserve the meaning, legal references and technical terms
(e.g. "high-risk AI system", "conformity assessment"). Do not add or
remove content. Return ONLY the translation.
"""


_TRANSLATE_FROM_EN_SYSTEM_TEMPLATE = """\
You are a professional translator. Translate the text into {language_name}.
Faithfully preserve the meaning, legal references and technical terms.
Do not add or remove content. Return ONLY the translation.
"""


class _Translation(BaseModel):
    """Schema of the LLM response for translation."""
    text: str


def detect_language(text: str, llm: LLMClient) -> str:
    """Detect the ISO 639-1 language code of the input.

    Returns `"en"` as a fallback if detection fails.
    """
    try:
        result = llm.structured_complete(
            text,
            _LanguageDetection,
            system_prompt=_LANGUAGE_DETECTION_SYSTEM,
        )
        code = result.code.strip().lower()
        # Normalize: some LLMs return "it-IT" or "IT"
        code = code.split("-")[0].split("_")[0]
        return code
    except LLMError as exc:
        logger.warning("Language not detected (%s); assuming 'en'", exc)
        return "en"


def translate_to_english(text: str, llm: LLMClient) -> str:
    """Translate `text` into English. On failure, return the original."""
    try:
        result = llm.structured_complete(
            text,
            _Translation,
            system_prompt=_TRANSLATE_TO_EN_SYSTEM,
        )
        return result.text
    except LLMError as exc:
        logger.warning("Translation to English failed (%s); using original", exc)
        return text


def translate_from_english(
    text: str, target_code: str, llm: LLMClient
) -> str:
    """Translate `text` from English to `target_code`.

    If `target_code` is not in `languages.json` or is `en`, return the original.
    """
    if target_code == "en":
        return text
    language_name = SUPPORTED_LANGUAGES.get(target_code)
    if language_name is None:
        logger.info(
            "Language '%s' not supported by languages.json; no translation",
            target_code,
        )
        return text

    system_prompt = _TRANSLATE_FROM_EN_SYSTEM_TEMPLATE.format(
        language_name=language_name
    )
    try:
        result = llm.structured_complete(
            text,
            _Translation,
            system_prompt=system_prompt,
        )
        return result.text
    except LLMError as exc:
        logger.warning(
            "Translation to '%s' failed (%s); using English",
            target_code,
            exc,
        )
        return text