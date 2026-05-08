"""NDJSON wire format v1 for ``chief serve`` ↔ clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from chief.domain import Episode

Target = Literal["orchestrator", "subagent"]


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Validated chat request parsed from one JSON line."""

    session_id: str
    target: Target
    text: str
    provider: str | None


def parse_request_line(line: str) -> ChatRequest:
    r"""Parse and validate protocol version 1 JSON from a single line.

    Expected keys: ``v`` (must be ``1``), ``session_id`` (optional), ``target``
    (``orchestrator`` | ``subagent``), ``text`` (non-empty task string),
    ``provider`` (required when ``target`` is ``subagent``: registry id or ``fake``).

    Args:
        line: Raw line without trailing ``\n`` or with it stripped by caller.

    Returns:
        Normalized :class:`ChatRequest`.

    Raises:
        ValueError: On invalid JSON, wrong version, missing fields, or bad enums.
    """
    raw = line.strip()
    if not raw:
        raise ValueError("empty line")
    try:
        obj: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json:{exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("request must be a JSON object")
    if obj.get("v") != 1:
        raise ValueError("unsupported v; expected 1")
    sid = obj.get("session_id", "default")
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError("session_id must be a non-empty string")
    sid = sid.strip()
    target = obj.get("target", "orchestrator")
    if target not in ("orchestrator", "subagent"):
        raise ValueError("target must be orchestrator or subagent")
    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    prov_raw = obj.get("provider", None)
    provider: str | None
    if prov_raw is None or prov_raw == "":
        provider = None
    elif isinstance(prov_raw, str):
        provider = prov_raw.strip().lower() or None
    else:
        raise ValueError("provider must be a string or null")
    if target == "subagent" and not provider:
        raise ValueError("target=subagent requires provider (registry id or fake)")
    if target == "orchestrator" and provider is not None:
        raise ValueError("target=orchestrator must omit provider (use server default)")
    return ChatRequest(session_id=sid, target=target, text=text.strip(), provider=provider)


def response_ok(session_id: str, episode: Episode) -> dict[str, Any]:
    """Build a successful v1 response envelope including episode summary.

    Args:
        session_id: Echo of the logical session id from the request.
        episode: Finished episode aggregate.

    Returns:
        JSON-serializable dict with ``v``, ``ok``, ``session_id``, ``episode``.
    """
    return {
        "v": 1,
        "ok": True,
        "session_id": session_id,
        "episode": {
            "id": episode.id,
            "status": episode.status.value,
            "artifact": episode.artifact,
            "ticks": len(episode.ticks),
        },
        "error": None,
    }


def response_error(message: str) -> dict[str, Any]:
    """Build a v1 error response (no episode).

    Args:
        message: Human-readable error for the client.

    Returns:
        JSON-serializable dict with ``ok: false`` and ``error`` string.
    """
    return {
        "v": 1,
        "ok": False,
        "session_id": None,
        "episode": None,
        "error": message,
    }
