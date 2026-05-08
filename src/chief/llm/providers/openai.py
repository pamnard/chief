"""OpenAI vendor **Chat Completions** API (:class:`~chief.brain.Brain`).

Self-contained: does not import :mod:`chief.llm.providers.custom_chat_completions`.
Uses :class:`~chief.config.runtime.RuntimeConfig` built at process entry.

Wire schema: :mod:`chief.llm.schema.openai_chat_completions`.
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
class OpenAiChatCompletionsBrain:
    """Planner for the **vendor** OpenAI Chat Completions HTTP API.

    Uses the same JSON wire as :class:`~chief.llm.providers.custom_chat_completions.CustomChatCompletionsBrain`
    but reads defaults from the canonical ``openai`` registry row (including official base URL when
    the registry leaves ``base_url`` empty, resolved earlier in config loading).

    Attributes:
        endpoint: Vendor API base (typically ``https://api.openai.com/v1``) and API key.
        model: OpenAI model id for the request body.
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
                f"OpenAiChatCompletionsBrain requires OPENAI_CHAT_COMPLETIONS wire, "
                f"got {self.endpoint.wire_format!r}"
            )

    @classmethod
    def from_runtime(cls, rt: RuntimeConfig) -> OpenAiChatCompletionsBrain:
        """Build from the canonical ``openai`` slice and planner prompt on ``rt``.

        Args:
            rt: Snapshot from :func:`chief.config.runtime.build_runtime_config`.

        Returns:
            Frozen brain whose ``api_base`` and model settings mirror ``rt.openai``.
        """
        o = rt.openai
        api_base = o.vendor_api_base.rstrip("/")
        endpoint = ProviderEndpoint(
            wire_format=ChatWireFormat.OPENAI_CHAT_COMPLETIONS,
            api_base=api_base,
            api_key=o.api_key,
        )
        rec = rt.providers_by_id["openai"]
        api_model, jm = openai_wire_model_and_json_mode(rt, rec)
        return cls(
            endpoint=endpoint,
            model=ModelRef(id=api_model),
            timeout_seconds=o.timeout_seconds,
            request_json_mode=jm,
            system_prompt=rt.system_prompt,
        )

    async def reason(self, memory: MemorySession, task: str) -> Intent:
        """POST ``.../chat/completions`` once and parse the assistant message into an intent.

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
