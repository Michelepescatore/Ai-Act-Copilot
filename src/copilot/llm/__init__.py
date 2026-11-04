"""Entry point for the LLM package.

Exposes only `LLMClient` and `LLMError`: callers do
`from copilot.llm import LLMClient` without knowing the internal layout.
"""

from copilot.llm.client import LLMClient, LLMError

__all__ = ["LLMClient", "LLMError"]
