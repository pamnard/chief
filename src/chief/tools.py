"""Built-in tool implementations for the v0 prototype.

Each tool maps argument dicts to :class:`~chief.domain.Observation`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from chief.domain import Observation

ToolFn = Callable[[dict[str, Any]], Observation]


def tool_noop(args: dict[str, Any]) -> Observation:
    """Return a successful observation echoing arguments.

    Args:
        args: Arbitrary JSON-like arguments.

    Returns:
        Observation with ``ok=True`` and diagnostic payload.
    """
    return Observation(ok=True, payload={"tool": "noop", "args": args})


def tool_echo(args: dict[str, Any]) -> Observation:
    """Echo the ``text`` field when present.

    Args:
        args: Expected key ``text`` for message body.

    Returns:
        Successful observation containing echoed text under ``echo``.
    """
    text = args.get("text", "")
    return Observation(ok=True, payload={"tool": "echo", "echo": text})


def tool_broken(args: dict[str, Any]) -> Observation:
    """Simulate a failing tool for tests and FakeBrain replan paths.

    Args:
        args: Ignored.

    Returns:
        Observation with ``ok=False`` and an error marker payload.
    """
    return Observation(ok=False, payload={"tool": "broken", "error": "simulated failure"})


def build_registry() -> dict[str, ToolFn]:
    """Construct the default tool name → callable registry.

    Returns:
        Mapping containing ``noop``, ``echo``, and ``broken``.
    """
    return {
        "noop": tool_noop,
        "echo": tool_echo,
        "broken": tool_broken,
    }
