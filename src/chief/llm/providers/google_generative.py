""":class:`~chief.brain.Brain` for Google Generative Language ``generateContent``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from chief.domain import Intent
from chief.llm.errors import ChatCompletionTransportError
from chief.llm.intent_json import parse_intent_from_model_json
from chief.llm.planner_context import serialize_episode_context
from chief.llm.schema import google_generative as gg
from chief.llm.types import ChatWireFormat, ModelRef, ProviderEndpoint
from chief.memory import MemorySession
from chief.config import RuntimeConfig


@dataclass(frozen=True)
class GeminiGenerateContentBrain:
    """Brain using Gemini ``generateContent`` (API key in query string)."""

    endpoint: ProviderEndpoint
    model: ModelRef
    timeout_seconds: float
    system_prompt: str

    def __post_init__(self) -> None:
        if self.endpoint.wire_format is not ChatWireFormat.GOOGLE_GENERATIVE_AI:
            raise ValueError(
                f"GeminiGenerateContentBrain requires GOOGLE_GENERATIVE_AI, got {self.endpoint.wire_format!r}"
            )
        if not self.endpoint.api_key or not self.endpoint.api_key.strip():
            raise ValueError("Gemini requires api_key on ProviderEndpoint (used as ?key=)")

    @classmethod
    def from_runtime(cls, rt: RuntimeConfig) -> GeminiGenerateContentBrain:
        g = rt.gemini
        base = g.vendor_api_base.rstrip("/")
        endpoint = ProviderEndpoint(
            wire_format=ChatWireFormat.GOOGLE_GENERATIVE_AI,
            api_base=base,
            api_key=g.api_key,
        )
        return cls(
            endpoint=endpoint,
            model=ModelRef(id=g.model),
            timeout_seconds=g.timeout_seconds,
            system_prompt=rt.system_prompt,
        )

    async def reason(self, memory: MemorySession, task: str) -> Intent:
        """POST ``generateContent`` and map assistant text to ``Intent``."""
        user = serialize_episode_context(memory, task)
        contents = gg.build_contents(user)
        payload = gg.build_request_body(contents=contents, system_instruction=self.system_prompt)
        body = json.dumps(payload).encode("utf-8")
        path_url = gg.generate_content_url(self.endpoint.api_base, self.model)
        url = f"{path_url}?key={self.endpoint.api_key}"
        headers = {"Content-Type": "application/json"}

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

        content = gg.extract_assistant_text(api_payload)
        return parse_intent_from_model_json(content)
