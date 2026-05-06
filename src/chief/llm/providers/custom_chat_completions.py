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


@dataclass(frozen=True)
class CustomChatCompletionsBrain:
    """Planner for a **custom** ``/v1`` base URL (OpenAI Chat Completions wire)."""

    endpoint: ProviderEndpoint
    model: ModelRef
    timeout_seconds: float
    request_json_mode: bool
    system_prompt: str

    def __post_init__(self) -> None:
        if self.endpoint.wire_format is not ChatWireFormat.OPENAI_CHAT_COMPLETIONS:
            raise ValueError(
                f"CustomChatCompletionsBrain requires OPENAI_CHAT_COMPLETIONS wire, "
                f"got {self.endpoint.wire_format!r}"
            )

    @classmethod
    def from_runtime(cls, rt: RuntimeConfig) -> CustomChatCompletionsBrain:
        c = rt.custom_llm
        api_base = c.base_url.rstrip("/")
        endpoint = ProviderEndpoint(
            wire_format=ChatWireFormat.OPENAI_CHAT_COMPLETIONS,
            api_base=api_base,
            api_key=c.api_key,
        )
        return cls(
            endpoint=endpoint,
            model=ModelRef(id=c.model),
            timeout_seconds=c.timeout_seconds,
            request_json_mode=c.json_mode,
            system_prompt=rt.system_prompt,
        )

    async def reason(self, memory: MemorySession, task: str) -> Intent:
        """POST ``chat/completions`` and map assistant text to ``Intent``."""
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
