"""LLM integration: wire schemas, endpoints, and :class:`~chief.brain.Brain` adapters.

Layout:

* ``chief.llm.types`` — :class:`~chief.llm.types.ChatWireFormat`, :class:`~chief.llm.types.ProviderEndpoint`,
  :class:`~chief.llm.types.ModelRef` (where to call, which protocol, which model id).
* ``chief.llm.schema.*`` — vendor-neutral **wire schemas**: build URL/body, parse assistant text for one API shape.
* ``chief.llm.planner_context`` — **shared** user/system text for the planner (same across brains).
* ``chief.llm.providers.*`` — planners are built with :meth:`~chief.llm.providers.openai.OpenAiChatCompletionsBrain.from_runtime`
  from a process-wide :class:`~chief.config.runtime.RuntimeConfig` (see :func:`~chief.config.runtime.build_runtime_config`).
  schema ``schema.openai_chat_completions`` for the chat-completions path.
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
