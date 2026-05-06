""":class:`~chief.brain.Brain` for Anthropic Messages API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from chief.domain import Intent
from chief.llm.errors import ChatCompletionTransportError
from chief.llm.intent_json import parse_intent_from_model_json
from chief.llm.planner_context import serialize_episode_context
from chief.llm.schema import anthropic_messages as am
from chief.llm.types import ChatWireFormat, ModelRef, ProviderEndpoint
from chief.memory import MemorySession
from chief.config import RuntimeConfig


@dataclass(frozen=True)
class AnthropicMessagesBrain:
    """Brain using Anthropic ``/v1/messages``."""

    endpoint: ProviderEndpoint
    model: ModelRef
    timeout_seconds: float
    max_tokens: int
    api_version: str
    system_prompt: str

    def __post_init__(self) -> None:
        if self.endpoint.wire_format is not ChatWireFormat.ANTHROPIC_MESSAGES:
            raise ValueError(
                f"AnthropicMessagesBrain requires ANTHROPIC_MESSAGES, got {self.endpoint.wire_format!r}"
            )
        if not self.endpoint.api_key or not self.endpoint.api_key.strip():
            raise ValueError("Anthropic requires api_key on ProviderEndpoint")

    @classmethod
    def from_runtime(cls, rt: RuntimeConfig) -> AnthropicMessagesBrain:
        a = rt.anthropic
        base = a.vendor_api_base.rstrip("/")
        endpoint = ProviderEndpoint(
            wire_format=ChatWireFormat.ANTHROPIC_MESSAGES,
            api_base=base,
            api_key=a.api_key,
        )
        return cls(
            endpoint=endpoint,
            model=ModelRef(id=a.model),
            timeout_seconds=a.timeout_seconds,
            max_tokens=a.max_tokens,
            api_version=a.api_version,
            system_prompt=rt.system_prompt,
        )

    async def reason(self, memory: MemorySession, task: str) -> Intent:
        """POST once to ``/v1/messages`` and map assistant text to ``Intent``."""
        user = serialize_episode_context(memory, task)
        messages = am.build_messages(user)
        payload = am.build_request_body(
            self.model,
            messages,
            system=self.system_prompt,
            max_tokens=self.max_tokens,
        )
        body = json.dumps(payload).encode("utf-8")
        url = am.messages_url(self.endpoint.api_base)
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.api_version,
            "x-api-key": self.endpoint.api_key or "",
        }

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

        content = am.extract_assistant_text(api_payload)
        return parse_intent_from_model_json(content)
