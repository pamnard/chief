"""Custom gateway for the OpenAI **Chat Completions** JSON wire (your base URL, not the OpenAI vendor).

Self-contained: does not import :mod:`chief.llm.providers.openai`. Uses
:mod:`chief.llm.schema.openai_chat_completions` for URL/body.

Reads :class:`~chief.config.runtime.RuntimeConfig` (single load per process).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from chief.domain import Intent
from chief.llm.errors import ChatCompletionTransportError
from chief.llm.intent_json import parse_intent_from_model_json
from chief.llm.planner_context import serialize_episode_context
from chief.llm.schema import openai_chat_completions as occ
from chief.llm.types import ChatWireFormat, ModelRef, ProviderEndpoint
from chief.memory import MemorySession
from chief.config import RuntimeConfig
from chief.config.runtime import openai_wire_model_and_json_mode


@dataclass(frozen=True)
class CustomChatCompletionsBrain:
    """Planner for a **custom** gateway speaking the OpenAI Chat Completions JSON wire.

    Unlike :class:`~chief.llm.providers.openai.OpenAiChatCompletionsBrain`, this class does not
    assume the OpenAI vendor host; ``api_base`` comes from the user registry (e.g. local Ollama).

    Attributes:
        endpoint: Target base URL and optional bearer token (OpenAI-shaped wire).
        model: Model id passed through to the request body.
        timeout_seconds: Per-request ``httpx`` timeout budget.
        request_json_mode: Chat Completions ``response_format`` JSON object mode when supported.
        system_prompt: Rendered planner instructions (allowed tools, JSON shapes).
    """

    endpoint: ProviderEndpoint
    model: ModelRef
    timeout_seconds: float
    request_json_mode: bool
    system_prompt: str

    def __post_init__(self) -> None:
        """Reject endpoints that do not use the Chat Completions wire enum.

        Raises:
            ValueError: If ``endpoint.wire_format`` is not ``OPENAI_CHAT_COMPLETIONS``.
        """
        if self.endpoint.wire_format is not ChatWireFormat.OPENAI_CHAT_COMPLETIONS:
            raise ValueError(
                f"CustomChatCompletionsBrain requires OPENAI_CHAT_COMPLETIONS wire, "
                f"got {self.endpoint.wire_format!r}"
            )

    @classmethod
    def from_runtime(cls, rt: RuntimeConfig) -> CustomChatCompletionsBrain:
        """Build from the canonical ``custom_llm`` slice and planner prompt on ``rt``.

        Args:
            rt: Snapshot from :func:`chief.config.runtime.build_runtime_config` (must include
                registry id ``custom_llm`` with kind ``custom_llm``).

        Returns:
            Frozen brain whose ``api_base`` and model settings mirror ``rt.custom_llm``.
        """
        c = rt.custom_llm
        api_base = c.base_url.rstrip("/")
        endpoint = ProviderEndpoint(
            wire_format=ChatWireFormat.OPENAI_CHAT_COMPLETIONS,
            api_base=api_base,
            api_key=c.api_key,
        )
        rec = rt.providers_by_id["custom_llm"]
        api_model, jm = openai_wire_model_and_json_mode(rt, rec)
        return cls(
            endpoint=endpoint,
            model=ModelRef(id=api_model),
            timeout_seconds=c.timeout_seconds,
            request_json_mode=jm,
            system_prompt=rt.system_prompt,
        )

    async def reason(self, memory: MemorySession, task: str) -> Intent:
        """POST ``.../chat/completions`` once and parse the assistant message into an intent.

        Builds messages via :mod:`chief.llm.schema.openai_chat_completions`, sends JSON with
        optional ``Authorization: Bearer``, then decodes the assistant string through
        :func:`chief.llm.intent_json.parse_intent_from_model_json`.

        Args:
            memory: Episode memory serialized into the user content.
            task: Task string for this reasoning step.

        Returns:
            A :class:`~chief.domain.ToolIntent` or :class:`~chief.domain.FinalIntent`.

        Raises:
            ChatCompletionTransportError: On non-success HTTP status or transport errors from httpx.
            IntentPayloadError: If the assistant content is not valid planner JSON.
        """
        user = serialize_episode_context(memory, task)
        messages = occ.build_messages(self.system_prompt, user)
        payload = occ.build_request_body(
            self.model,
            messages,
            json_mode=self.request_json_mode,
        )
        body = json.dumps(payload).encode("utf-8")
        url = occ.chat_completions_url(self.endpoint.api_base)
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"

        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, content=body, headers=headers)
                resp.raise_for_status()
                api_payload: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            reason = exc.response.reason_phrase
            raise ChatCompletionTransportError(f"http_{code}:{reason}") from exc
        except httpx.RequestError as exc:
            raise ChatCompletionTransportError(f"network:{exc!s}") from exc

        content = occ.extract_assistant_text(api_payload)
        return parse_intent_from_model_json(content)
