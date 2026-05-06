"""Domain types: wire format, provider endpoint, model reference.

* **ChatWireFormat** — which message encoding / API contract we speak (not a vendor).
* **ProviderEndpoint** — network location + credentials for *one* deployment.
* **ModelRef** — model id string as understood by that provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChatWireFormat(str, Enum):
    """Supported wire protocols for LLM-backed :class:`~chief.brain.Brain` adapters."""

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GOOGLE_GENERATIVE_AI = "google_generative_ai"


@dataclass(frozen=True)
class ProviderEndpoint:
    """HTTP(S) endpoint that accepts requests for a given wire format.

    Attributes:
        wire_format: Protocol used when encoding requests and decoding responses.
        api_base: Base URL for that wire (see codec module for trailing path rules).
        api_key: Optional secret (bearer for OpenAI/Anthropic; for Gemini often passed
            in query string by the provider adapter instead of this field).
    """

    wire_format: ChatWireFormat
    api_base: str
    api_key: str | None = None


@dataclass(frozen=True)
class ModelRef:
    """Identifies a concrete model inside a provider's catalog.

    Attributes:
        id: Provider-native model string (e.g. ``gpt-4o-mini``, ``llama3.2``).
    """

    id: str
