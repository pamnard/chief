"""LLM integration: wire schemas, endpoints, and :class:`~chief.brain.Brain` adapters.

Layout:

* ``chief.llm.types`` — :class:`~chief.llm.types.ChatWireFormat`, :class:`~chief.llm.types.ProviderEndpoint`,
  :class:`~chief.llm.types.ModelRef` (where to call, which protocol, which model id).
* ``chief.llm.schema.*`` — vendor-neutral **wire schemas**: build URL/body, parse assistant text for one API shape.
* ``chief.llm.planner_context`` — **shared** user/system text for the planner (same across brains).
* ``chief.llm.providers.*`` — HTTP brains expose ``from_runtime`` and take a process-wide
  :class:`~chief.config.runtime.RuntimeConfig` from :func:`~chief.config.runtime.build_runtime_config`.
"""

from chief.llm.errors import ChatCompletionTransportError, IntentPayloadError, LlmError
from chief.llm.intent_json import parse_intent_from_model_json
from chief.llm.providers import (
    AnthropicMessagesBrain,
    CustomChatCompletionsBrain,
    GeminiGenerateContentBrain,
    OpenAiChatCompletionsBrain,
)
from chief.llm.types import ChatWireFormat, ModelRef, ProviderEndpoint

__all__ = [
    "AnthropicMessagesBrain",
    "ChatCompletionTransportError",
    "ChatWireFormat",
    "CustomChatCompletionsBrain",
    "GeminiGenerateContentBrain",
    "IntentPayloadError",
    "LlmError",
    "ModelRef",
    "OpenAiChatCompletionsBrain",
    "ProviderEndpoint",
    "parse_intent_from_model_json",
]
