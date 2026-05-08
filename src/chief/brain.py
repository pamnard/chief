"""Planner / brain abstraction.

The :class:`Brain` protocol isolates LLM or scripted planners from the engine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from chief.config import RuntimeConfig
from chief.domain import FinalIntent, Intent, ToolIntent
from chief.memory import MemorySession


@runtime_checkable
class Brain(Protocol):
    """Produces the next intent given session memory and task text."""

    async def reason(self, memory: MemorySession, task: str) -> Intent:
        """Return the next planner decision.

        Args:
            memory: Observations from prior acts in this episode.
            task: Original task string from the trigger.

        Returns:
            Either a tool intent or a terminal ``FinalIntent``.
        """
        ...


class FakeBrain:
    """Deterministic scripted planner for tests and offline demos.

    The first and second tool names are taken from ``runtime`` (``[fake_brain]`` in merged
    configuration) so the scenario stays configurable without hard-coded tool strings in code.

    Attributes:
        _first: Tool name invoked on the first ``reason`` call when there is no observation.
        _second: Tool name invoked after a failed observation.
    """

    def __init__(self, runtime: RuntimeConfig) -> None:
        """Capture fake-brain tool names from a :class:`~chief.config.runtime.RuntimeConfig`.

        Args:
            runtime: Process configuration snapshot from :func:`~chief.config.runtime.build_runtime_config`.
        """
        self._first = runtime.fake_brain_first_tool
        self._second = runtime.fake_brain_second_tool

    async def reason(self, memory: MemorySession, task: str) -> Intent:
        """Drive the scripted multi-step scenario based on prior observations.

        Args:
            memory: Session containing zero or more observations.
            task: Task text forwarded to the second tool after the simulated failure.

        Returns:
            ``ToolIntent`` or ``FinalIntent`` according to the scripted state machine.
        """
        last = memory.last_observation()
        if last is None:
            return ToolIntent(self._first, {})
        if not last.ok:
            return ToolIntent(self._second, {"text": task or "ok"})
        return FinalIntent(f"done: {task!r}" if task else "done")
