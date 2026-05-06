"""LLM integration: wire formats, provider endpoints, and Brain adapters."""

from chief.llm.errors import ChatCompletionTransportError, IntentPayloadError, LlmError
from chief.llm.intent_json import parse_intent_from_model_json
from chief.llm.providers.http_chat_completions import HttpChatCompletionsBrain, OpenAIChatBrain
from chief.llm.types import ChatWireFormat, ModelRef, ProviderEndpoint

__all__ = [
    "ChatCompletionTransportError",
    "ChatWireFormat",
    "HttpChatCompletionsBrain",
    "IntentPayloadError",
    "LlmError",
    "ModelRef",
    "OpenAIChatBrain",
    "ProviderEndpoint",
    "parse_intent_from_model_json",
]
