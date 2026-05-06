"""Codec for Google AI ``:generateContent`` REST (Generative Language API wire).

Uses ``v1beta`` style paths; API key is usually supplied as ``?key=`` on the URL,
not as ``Authorization`` (handled in the HTTP provider).
"""

from __future__ import annotations

from typing import Any

from chief.llm.errors import ChatCompletionTransportError
from chief.llm.types import ChatWireFormat, ModelRef


def wire_format() -> ChatWireFormat:
    """Return the enum tag represented by this module."""
    return ChatWireFormat.GOOGLE_GENERATIVE_AI


def generate_content_url(api_base: str, model: ModelRef) -> str:
    """Build ``POST`` URL ``{api_base}/models/{model}:generateContent``.

    Args:
        api_base: e.g. ``https://generativelanguage.googleapis.com/v1beta``.
        model: Model id (e.g. ``gemini-1.5-flash``).
    """
    root = api_base.rstrip("/")
    mid = model.id
    if mid.startswith("models/"):
        mid = mid[len("models/") :]
    return f"{root}/models/{mid}:generateContent"


def build_contents(user_text: str) -> list[dict[str, Any]]:
    """Single user turn in Gemini ``contents`` shape."""
    return [{"role": "user", "parts": [{"text": user_text}]}]


def build_request_body(
    *,
    contents: list[dict[str, Any]],
    system_instruction: str | None,
) -> dict[str, Any]:
    """Assemble JSON body for ``generateContent``."""
    body: dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    return body


def extract_assistant_text(response_json: dict[str, Any]) -> str:
    """Read first candidate text part from a ``generateContent`` response."""
    try:
        cands = response_json["candidates"]
        parts = cands[0]["content"]["parts"]
        text = parts[0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatCompletionTransportError(f"bad_response_shape:{response_json!r}") from exc
    if not isinstance(text, str):
        raise ChatCompletionTransportError("assistant_content_not_string")
    return text
