"""Parse planner JSON embedded in assistant text into domain intents."""

from __future__ import annotations

import json
import re
from typing import Any

from chief.domain import FinalIntent, Intent, ToolIntent
from chief.llm.errors import IntentPayloadError


def _strip_json_fence(raw: str) -> str:
    """Remove optional Markdown ```json fences from model output.

    Args:
        raw: Raw assistant string.

    Returns:
        Inner JSON text suitable for ``json.loads``.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_intent_from_model_json(raw: str) -> Intent:
    """Parse planner JSON from assistant text into an ``Intent``.

    Expected shapes::

        {"intent_type": "tool", "tool": "echo", "args": {...}}
        {"intent_type": "final", "message": "..."}

    Args:
        raw: Assistant message content (possibly fenced).

    Returns:
        ``ToolIntent`` or ``FinalIntent``.

    Raises:
        IntentPayloadError: If JSON is invalid or required fields are missing.
    """
    text = _strip_json_fence(raw)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IntentPayloadError(f"invalid_json:{exc}") from exc
    if not isinstance(data, dict):
        raise IntentPayloadError("intent_payload_not_object")
    kind = data.get("intent_type")
    if kind == "final":
        msg = data.get("message")
        if not isinstance(msg, str) or not msg.strip():
            raise IntentPayloadError("final_intent_missing_message")
        return FinalIntent(msg.strip())
    if kind == "tool":
        name = data.get("tool")
        if not isinstance(name, str) or not name.strip():
            raise IntentPayloadError("tool_intent_missing_tool")
        args = data.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise IntentPayloadError("tool_intent_args_not_object")
        return ToolIntent(name.strip(), dict(args))
    raise IntentPayloadError(f"unknown_intent_type:{kind!r}")
