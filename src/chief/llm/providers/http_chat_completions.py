"""HTTP client for :class:`~chief.llm.types.ChatWireFormat.OPENAI_CHAT_COMPLETIONS`.

Combines a :class:`~chief.llm.types.ProviderEndpoint`, a :class:`~chief.llm.types.ModelRef`,
and the wire codec in :mod:`chief.llm.formats.openai_chat_completions`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from chief.brain import Brain
from chief.domain import Intent
from chief.llm.errors import ChatCompletionTransportError
from chief.llm.formats import openai_chat_completions as occ
from chief.llm.intent_json import parse_intent_from_model_json
from chief.llm.types import ChatWireFormat, ModelRef, ProviderEndpoint
from chief.memory import MemorySession


def _serialize_context(memory: MemorySession, task: str) -> str:
    """Build user-role text listing task and observations.

    Args:
        memory: Session observations from prior acts.
        task: Original task string.

    Returns:
        Multi-line user message body.
    """
    lines = [f"Task: {task}"]
    for i, obs in enumerate(memory.observations, start=1):
        payload = json.dumps(obs.payload, ensure_ascii=False)
        lines.append(f"Observation {i}: ok={obs.ok} payload={payload}")
    return "\n".join(lines)


def _system_prompt(allowed_tools: tuple[str, ...]) -> str:
    """Return instructions forcing a single JSON object as the planner reply.

    Args:
        allowed_tools: Tool names accepted by policy in v0.

    Returns:
        System message content.
    """
    tools = ", ".join(sorted(allowed_tools))
    return (
        "You are the Reason phase of a single-step agent loop. "
        "Reply with ONE JSON object only, no prose. "
        f"Allowed tool names: {tools}. "
        'Shape A: {"intent_type":"tool","tool":"<name>","args":{...}} '
        'Shape B: {"intent_type":"final","message":"<user-facing summary>"}. '
        "Use final when the task is done or impossible. "
        'Use echo with args {"text": "..."} to return text to the user. '
        "Prefer small args objects."
    )


@dataclass(frozen=True)
class HttpChatCompletionsBrain:
    """Brain using Chat Completions wire format over HTTP.

    Attributes:
        endpoint: Provider URL + auth; ``wire_format`` must be
            :attr:`~chief.llm.types.ChatWireFormat.OPENAI_CHAT_COMPLETIONS`.
        model: Which provider model to query.
        timeout_seconds: Socket read timeout for each HTTP call.
        allowed_tools: Declared tool names for the system prompt (policy alignment).
        request_json_mode: Pass ``response_format`` when the server supports it.
    """

    endpoint: ProviderEndpoint
    model: ModelRef
    timeout_seconds: float = 120.0
    allowed_tools: tuple[str, ...] = ("noop", "echo", "broken")
    request_json_mode: bool = False

    def __post_init__(self) -> None:
        if self.endpoint.wire_format is not ChatWireFormat.OPENAI_CHAT_COMPLETIONS:
            raise ValueError(
                f"HttpChatCompletionsBrain requires OPENAI_CHAT_COMPLETIONS, got {self.endpoint.wire_format!r}"
            )

    @classmethod
    def from_env(cls) -> HttpChatCompletionsBrain:
        """Build from ``CHIEF_LLM_*`` environment variables.

        - ``CHIEF_LLM_BASE_URL`` — ``.../v1`` (default ``http://127.0.0.1:11434/v1``).
        - ``CHIEF_LLM_MODEL`` — model id (default ``llama3.2``).
        - ``CHIEF_LLM_API_KEY`` — optional bearer token.
        - ``CHIEF_LLM_TIMEOUT`` — seconds (default ``120``).
        - ``CHIEF_LLM_JSON_MODE`` — ``1`` / ``true`` for JSON object mode request.

        Returns:
            Configured brain instance.
        """
        base = os.environ.get("CHIEF_LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        model_id = os.environ.get("CHIEF_LLM_MODEL", "llama3.2")
        key = os.environ.get("CHIEF_LLM_API_KEY") or None
        if key is not None and not key.strip():
            key = None
        timeout = float(os.environ.get("CHIEF_LLM_TIMEOUT", "120"))
        json_mode = os.environ.get("CHIEF_LLM_JSON_MODE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        endpoint = ProviderEndpoint(
            wire_format=ChatWireFormat.OPENAI_CHAT_COMPLETIONS,
            api_base=base,
            api_key=key,
        )
        return cls(
            endpoint=endpoint,
            model=ModelRef(id=model_id),
            timeout_seconds=timeout,
            request_json_mode=json_mode,
        )

    def reason(self, memory: MemorySession, task: str) -> Intent:
        """POST once to ``chat/completions`` and map assistant text to ``Intent``.

        Args:
            memory: Prior observations in this episode.
            task: Task text from the trigger.

        Returns:
            Parsed ``Intent``.

        Raises:
            ChatCompletionTransportError: On HTTP or envelope errors.
            IntentPayloadError: When assistant text is not valid planner JSON.
        """
        system = _system_prompt(self.allowed_tools)
        user = _serialize_context(memory, task)
        messages = occ.build_messages(system, user)
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

        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                api_payload: dict[str, Any] = json.load(resp)
        except urllib.error.HTTPError as exc:
            raise ChatCompletionTransportError(f"http_{exc.code}:{exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ChatCompletionTransportError(f"network:{exc.reason}") from exc
        except TimeoutError as exc:
            raise ChatCompletionTransportError("timeout") from exc

        content = occ.extract_assistant_text(api_payload)
        return parse_intent_from_model_json(content)


# Backwards-compatible alias for early imports / docs.
OpenAIChatBrain = HttpChatCompletionsBrain
