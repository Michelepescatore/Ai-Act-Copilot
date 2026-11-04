"""Rilevamento lingua e traduzione.

Estrae la lingua dell'input dell'utente via LLM, poi traduce in inglese prima
della pipeline e di nuovo nella lingua originale prima del report.

Se la lingua non è in `language.json` o è già inglese, la traduzione viene
saltata e il flusso usa l'input originale.
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
    """Carica il mapping codice → nome da `language.json`."""
    with LANGUAGE_FILE.open(encoding="utf-8") as f:
        return json.load(f)


# Cache del mapping a livello di modulo (il file non cambia a runtime)
SUPPORTED_LANGUAGES = _load_supported_languages()


class _LanguageDetection(BaseModel):
    """Schema della risposta dell'LLM per il rilevamento lingua."""
    code: str  # ISO 639-1, es. "it", "en", "ja"


_LANGUAGE_DETECTION_SYSTEM = """\
Sei un rilevatore di lingua. Data una stringa di testo, restituisci il codice
ISO 639-1 della lingua principale (due lettere, es. "it", "en", "ja", "ar").
Se il testo è ambiguo, scegli la lingua dominante.
"""


_TRANSLATE_TO_EN_SYSTEM = """\
Sei un traduttore professionista. Traduci il testo dell'utente in inglese.
Mantieni fedelmente il significato, i riferimenti normativi e i termini tecnici
(es. "high-risk AI system", "conformity assessment"). Non aggiungere né
rimuovere contenuto. Restituisci SOLO la traduzione.
"""


_TRANSLATE_FROM_EN_SYSTEM_TEMPLATE = """\
Sei un traduttore professionista. Traduci il testo in {language_name}.
Mantieni fedelmente il significato, i riferimenti normativi e i termini tecnici.
Non aggiungere né rimuovere contenuto. Restituisci SOLO la traduzione.
"""


class _Translation(BaseModel):
    """Schema della risposta dell'LLM per la traduzione."""
    text: str


def detect_language(text: str, llm: LLMClient) -> str:
    """Rileva il codice lingua ISO 639-1 dell'input.

    Restituisce `"en"` come fallback se il rilevamento fallisce.
    """
    try:
        result = llm.structured_complete(
            text,
            _LanguageDetection,
            system_prompt=_LANGUAGE_DETECTION_SYSTEM,
        )
        code = result.code.strip().lower()
        # Normalizza: alcuni LLM restituiscono "it-IT" o "IT"
        code = code.split("-")[0].split("_")[0]
        return code
    except LLMError as exc:
        logger.warning("Lingua non rilevata (%s); assumo 'en'", exc)
        return "en"


def translate_to_english(text: str, llm: LLMClient) -> str:
    """Traduce `text` in inglese. Se fallisce, restituisce l'originale."""
    try:
        result = llm.structured_complete(
            text,
            _Translation,
            system_prompt=_TRANSLATE_TO_EN_SYSTEM,
        )
        return result.text
    except LLMError as exc:
        logger.warning("Traduzione in inglese fallita (%s); uso originale", exc)
        return text


def translate_from_english(
    text: str, target_code: str, llm: LLMClient
) -> str:
    """Traduce `text` dall'inglese a `target_code`.

    Se `target_code` non è in `language.json` o è `en`, restituisce l'originale.
    """
    if target_code == "en":
        return text
    language_name = SUPPORTED_LANGUAGES.get(target_code)
    if language_name is None:
        logger.info(
            "Lingua '%s' non supportata da language.json; nessuna traduzione",
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
            "Traduzione verso '%s' fallita (%s); uso inglese",
            target_code,
            exc,
        )
        return text