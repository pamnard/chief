"""Exceptions for LLM transport and planner payload parsing."""


class LlmError(RuntimeError):
    """Base class for LLM-related failures."""


class IntentPayloadError(LlmError):
    """Assistant output could not be parsed into an :class:`~chief.domain.Intent`."""


class ChatCompletionTransportError(LlmError):
    """HTTP failure or response shape incompatible with the wire format."""
