"""Map CLI / IPC provider ids to concrete :class:`~chief.brain.Brain` implementations.

Registry rows use a ``kind`` (``custom_llm``, ``openai``, ``anthropic``, ``gemini``). Rows whose
``id`` matches the canonical id for that kind delegate to the corresponding ``from_runtime``;
additional rows with the same ``kind`` are built directly from :class:`~chief.config.providers_registry.ProviderRecord`.
"""

from __future__ import annotations

from chief.brain import Brain, FakeBrain
from chief.config import RuntimeConfig
from chief.config.runtime import openai_wire_model_and_json_mode
from chief.config.providers_registry import ProviderRecord
from chief.llm import (
    AnthropicMessagesBrain,
    CustomChatCompletionsBrain,
    GeminiGenerateContentBrain,
    OpenAiChatCompletionsBrain,
)
from chief.llm.types import ChatWireFormat, ModelRef, ProviderEndpoint


def _custom_from_record(record: ProviderRecord, rt: RuntimeConfig) -> CustomChatCompletionsBrain:
    """Build a custom Chat Completions brain from any ``custom_llm`` registry row.

    Used when ``record.id`` is not the canonical ``custom_llm`` id; wire format and codec match
    :class:`CustomChatCompletionsBrain` built via :meth:`CustomChatCompletionsBrain.from_runtime`.

    Args:
        record: Validated ``custom_llm`` row (``base_url``, ``model``, etc.).
        rt: Process configuration (model catalog resolution, system prompt).

    Returns:
        Frozen brain instance for one HTTP round-trip per :meth:`~CustomChatCompletionsBrain.reason`.
    """
    api_base = record.base_url.rstrip("/")
    endpoint = ProviderEndpoint(
        wire_format=ChatWireFormat.OPENAI_CHAT_COMPLETIONS,
        api_base=api_base,
        api_key=record.api_key,
    )
    api_model, jm = openai_wire_model_and_json_mode(rt, record)
    return CustomChatCompletionsBrain(
        endpoint=endpoint,
        model=ModelRef(id=api_model),
        timeout_seconds=record.timeout_seconds,
        request_json_mode=jm,
        system_prompt=rt.system_prompt,
    )


def _openai_vendor_from_record(record: ProviderRecord, rt: RuntimeConfig) -> OpenAiChatCompletionsBrain:
    """Build a vendor OpenAI Chat Completions brain from a non-canonical ``openai`` row.

    Args:
        record: Validated ``openai`` row; empty ``api_key`` becomes bearer-less requests.
        rt: Process configuration (model catalog resolution, system prompt).

    Returns:
        Frozen :class:`OpenAiChatCompletionsBrain` for the given endpoint and model.
    """
    api_base = record.base_url.rstrip("/")
    endpoint = ProviderEndpoint(
        wire_format=ChatWireFormat.OPENAI_CHAT_COMPLETIONS,
        api_base=api_base,
        api_key=record.api_key or "",
    )
    api_model, jm = openai_wire_model_and_json_mode(rt, record)
    return OpenAiChatCompletionsBrain(
        endpoint=endpoint,
        model=ModelRef(id=api_model),
        timeout_seconds=record.timeout_seconds,
        request_json_mode=jm,
        system_prompt=rt.system_prompt,
    )


def _anthropic_from_record(record: ProviderRecord, system_prompt: str) -> AnthropicMessagesBrain:
    """Build an Anthropic Messages brain from any ``anthropic`` registry row.

    Args:
        record: Row with ``base_url``, ``model``, ``api_key``, ``max_tokens``, ``api_version``.
        system_prompt: Rendered planner system text from merged config.

    Returns:
        Frozen :class:`AnthropicMessagesBrain`.

    Raises:
        ValueError: If ``max_tokens`` or ``api_version`` is missing on the record.
    """
    if record.max_tokens is None or record.api_version is None:
        raise ValueError("anthropic provider record requires max_tokens and api_version")
    base = record.base_url.rstrip("/")
    endpoint = ProviderEndpoint(
        wire_format=ChatWireFormat.ANTHROPIC_MESSAGES,
        api_base=base,
        api_key=record.api_key or "",
    )
    return AnthropicMessagesBrain(
        endpoint=endpoint,
        model=ModelRef(id=record.model),
        timeout_seconds=record.timeout_seconds,
        max_tokens=record.max_tokens,
        api_version=record.api_version,
        system_prompt=system_prompt,
    )


def _gemini_from_record(record: ProviderRecord, system_prompt: str) -> GeminiGenerateContentBrain:
    """Build a Gemini ``generateContent`` brain from any ``gemini`` registry row.

    Args:
        record: Row with ``base_url``, ``model``, ``api_key``, and timeout fields.
        system_prompt: Rendered planner system text from merged config.

    Returns:
        Frozen :class:`GeminiGenerateContentBrain`.
    """
    base = record.base_url.rstrip("/")
    endpoint = ProviderEndpoint(
        wire_format=ChatWireFormat.GOOGLE_GENERATIVE_AI,
        api_base=base,
        api_key=record.api_key or "",
    )
    return GeminiGenerateContentBrain(
        endpoint=endpoint,
        model=ModelRef(id=record.model),
        timeout_seconds=record.timeout_seconds,
        system_prompt=system_prompt,
    )


def _brain_from_record(record: ProviderRecord, rt: RuntimeConfig) -> Brain:
    """Dispatch on ``record.kind`` and id to construct the matching HTTP brain.

    Args:
        record: Entry from ``runtime.providers_by_id`` (already validated at load time).
        rt: Process configuration snapshot (slices, system prompt).

    Returns:
        A :class:`~chief.brain.Brain` implementation.

    Raises:
        ValueError: If ``record.kind`` is not one of the supported kinds.
    """
    if record.kind == "custom_llm":
        if record.id == "custom_llm":
            return CustomChatCompletionsBrain.from_runtime(rt)
        return _custom_from_record(record, rt)
    if record.kind == "openai":
        if record.id == "openai":
            return OpenAiChatCompletionsBrain.from_runtime(rt)
        return _openai_vendor_from_record(record, rt)
    if record.kind == "anthropic":
        if record.id == "anthropic":
            return AnthropicMessagesBrain.from_runtime(rt)
        return _anthropic_from_record(record, rt.system_prompt)
    if record.kind == "gemini":
        if record.id == "gemini":
            return GeminiGenerateContentBrain.from_runtime(rt)
        return _gemini_from_record(record, rt.system_prompt)
    raise ValueError(f"unknown provider kind: {record.kind!r}")


def select_brain(provider_id: str, runtime: RuntimeConfig) -> Brain:
    """Instantiate the planner implementation selected by ``provider_id``.

    Args:
        provider_id: ``fake`` for scripted planner, or an ``id`` from the provider registry.
        runtime: Process configuration snapshot.

    Returns:
        Concrete :class:`~chief.brain.Brain`.

    Raises:
        ValueError: If ``provider_id`` is unknown or not in the registry.
    """
    key = provider_id.strip().lower()
    if key == "fake":
        return FakeBrain(runtime)
    if key not in runtime.providers_by_id:
        known = ", ".join(sorted(runtime.providers_by_id)) if runtime.providers_by_id else "(empty)"
        raise ValueError(f"unknown provider: {provider_id!r} (known: fake, {known})")
    rec = runtime.providers_by_id[key]
    if not rec.enabled:
        raise ValueError(f"provider {provider_id!r} is disabled in the registry (enabled=false)")
    return _brain_from_record(rec, runtime)
