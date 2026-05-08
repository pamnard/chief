"""Built-in tool implementations for the v0 prototype.

Each tool maps argument dicts to :class:`~chief.domain.Observation`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from chief.domain import Observation
from chief.config import RuntimeConfig

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


_BUILTIN: dict[str, ToolFn] = {
    "noop": tool_noop,
    "echo": tool_echo,
    "broken": tool_broken,
}


def build_registry(runtime: RuntimeConfig) -> dict[str, ToolFn]:
    """Build a name-to-callable registry limited to configured planner tools.

    Args:
        runtime: Process configuration snapshot; only ``runtime.planner_allowed_tools`` names
            are included.

    Returns:
        Mapping of tool name to implementation for built-in tools.

    Raises:
        ValueError: If any configured name has no built-in implementation in this module.
    """
    out: dict[str, ToolFn] = {}
    for name in runtime.planner_allowed_tools:
        fn = _BUILTIN.get(name)
        if fn is None:
            raise ValueError(f"planner.allowed_tools contains unknown built-in tool: {name!r}")
        out[name] = fn
    return out
