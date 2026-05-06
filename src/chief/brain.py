"""Planner / brain abstraction.

The :class:`Brain` protocol isolates LLM or scripted planners from the engine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from chief.domain import FinalIntent, Intent, ToolIntent
from chief.memory import MemorySession
from chief.config import RuntimeConfig


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

    Tool sequence comes from :class:`~chief.config.runtime.RuntimeConfig` (built at process start).
    """

    def __init__(self, runtime: RuntimeConfig) -> None:
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
