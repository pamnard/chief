"""Codec for OpenAI ``POST /v1/chat/completions`` JSON (wire format).

This is the **message interchange shape** used by many providers (OpenAI, proxies,
Ollama OpenAI compatibility, etc.), not the OpenAI company itself.
"""

from __future__ import annotations

from typing import Any

from chief.llm.errors import ChatCompletionTransportError
from chief.llm.types import ChatWireFormat, ModelRef


def wire_format() -> ChatWireFormat:
    """Return the enum tag represented by this module."""
    return ChatWireFormat.OPENAI_CHAT_COMPLETIONS


def chat_completions_url(api_base_v1: str) -> str:
    """Build ``POST`` URL for Chat Completions.

    Args:
        api_base: Prefix ending with ``/v1`` (no trailing slash required).

    Returns:
        Full URL ending with ``chat/completions``.
    """
    root = api_base_v1.rstrip("/") + "/"
    return root + "chat/completions"


def build_messages(system_content: str, user_content: str) -> list[dict[str, str]]:
    """Create the ``messages`` array for a single-turn planner call.

    Args:
        system_content: System prompt text.
        user_content: User task + serialized observations.

    Returns:
        OpenAI-shaped role/content dicts.
    """
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_request_body(
    model: ModelRef,
    messages: list[dict[str, str]],
    *,
    json_mode: bool,
) -> dict[str, Any]:
    """Assemble the JSON body for ``/v1/chat/completions``.

    Args:
        model: Target model on the provider.
        messages: Chat turns (typically system + user).
        json_mode: When True, requests ``response_format`` JSON object mode if supported.

    Returns:
        Serializable request payload.
    """
    payload: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def extract_assistant_text(response_json: dict[str, Any]) -> str:
    """Read assistant ``content`` from a Chat Completions response envelope.

    Args:
        response_json: Parsed JSON body from the HTTP response.

    Returns:
        Assistant message string.

    Raises:
        ChatCompletionTransportError: If the envelope does not match expectations.
    """
    try:
        choices = response_json["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatCompletionTransportError(f"bad_response_shape:{response_json!r}") from exc
    if not isinstance(content, str):
        raise ChatCompletionTransportError("assistant_content_not_string")
    return content
