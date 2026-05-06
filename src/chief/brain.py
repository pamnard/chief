"""Planner / brain abstraction.

The :class:`Brain` protocol isolates LLM or scripted planners from the engine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from chief.domain import FinalIntent, Intent, ToolIntent
from chief.memory import MemorySession


@runtime_checkable
class Brain(Protocol):
    """Produces the next intent given session memory and task text."""

    def reason(self, memory: MemorySession, task: str) -> Intent:
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

    Sequence: call ``broken`` once, then ``echo`` with the task text, then emit a
    ``FinalIntent`` after a successful observation—exercising replanning on failure.
    """

    def reason(self, memory: MemorySession, task: str) -> Intent:
        """Drive the scripted multi-step scenario based on prior observations.

        Args:
            memory: Session containing zero or more observations.
            task: Task text forwarded to ``echo`` after the simulated failure.

        Returns:
            ``ToolIntent`` or ``FinalIntent`` according to the scripted state machine.
        """
        last = memory.last_observation()
        if last is None:
            return ToolIntent("broken", {})
        if not last.ok:
            return ToolIntent("echo", {"text": task or "ok"})
        return FinalIntent(f"done: {task!r}" if task else "done")
