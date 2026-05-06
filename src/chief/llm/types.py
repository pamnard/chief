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
    # Future: ANTHROPIC_MESSAGES, GOOGLE_GENERATIVE, ...


@dataclass(frozen=True)
class ProviderEndpoint:
    """HTTP(S) endpoint that accepts requests for a given wire format.

    Attributes:
        wire_format: Protocol used when encoding requests and decoding responses.
        api_base: URL prefix ending with ``/v1`` for OpenAI-compatible stacks.
        api_key: Optional bearer token (many local servers omit this).
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
