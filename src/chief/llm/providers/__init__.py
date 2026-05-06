"""Remote :class:`~chief.brain.Brain` adapters (one module per major vendor API).

Every brain here follows the same pattern: hold a :class:`~chief.llm.types.ProviderEndpoint` and
:class:`~chief.llm.types.ModelRef`, build request JSON with the matching ``chief.llm.schema.*`` codec,
fill prompts from :class:`~chief.config.runtime.RuntimeConfig` (serialized user text still via
:mod:`chief.llm.planner_context`), ``POST`` with ``httpx``, then
:class:`~chief.llm.intent_json.parse_intent_from_model_json` on the assistant string.

Sibling provider modules do not import each other.
"""

from chief.llm.providers.anthropic_messages import AnthropicMessagesBrain
from chief.llm.providers.custom_chat_completions import CustomChatCompletionsBrain
from chief.llm.providers.google_generative import GeminiGenerateContentBrain
from chief.llm.providers.openai import OpenAiChatCompletionsBrain

__all__ = [
    "AnthropicMessagesBrain",
    "CustomChatCompletionsBrain",
    "GeminiGenerateContentBrain",
    "OpenAiChatCompletionsBrain",
]
