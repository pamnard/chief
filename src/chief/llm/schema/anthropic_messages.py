"""Codec for Anthropic ``POST /v1/messages`` JSON (Messages API wire format).

See Anthropic API reference for ``messages`` resource; this module only shapes
request/response JSON, not vendor policy.
"""

from __future__ import annotations

from typing import Any

from chief.llm.errors import ChatCompletionTransportError
from chief.llm.types import ChatWireFormat, ModelRef


def wire_format() -> ChatWireFormat:
    """Return the enum tag represented by this module."""
    return ChatWireFormat.ANTHROPIC_MESSAGES


def messages_url(api_base_v1: str) -> str:
    """Build ``POST`` URL for Messages.

    Args:
        api_base_v1: Prefix such as ``https://api.anthropic.com/v1`` (no trailing slash).
    """
    return api_base_v1.rstrip("/") + "/messages"


def build_messages(user_content: str) -> list[dict[str, str]]:
    """Anthropic ``messages`` array for a single user turn (system is separate)."""
    return [{"role": "user", "content": user_content}]


def build_request_body(
    model: ModelRef,
    messages: list[dict[str, str]],
    *,
    system: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Assemble JSON body for ``/v1/messages``."""
    return {
        "model": model.id,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }


def extract_assistant_text(response_json: dict[str, Any]) -> str:
    """Read first text block from assistant ``content`` list."""
    try:
        content = response_json["content"]
        block = content[0]
        if block.get("type") != "text":
            raise ChatCompletionTransportError("anthropic_first_block_not_text")
        text = block["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatCompletionTransportError(f"bad_response_shape:{response_json!r}") from exc
    if not isinstance(text, str):
        raise ChatCompletionTransportError("assistant_content_not_string")
    return text
