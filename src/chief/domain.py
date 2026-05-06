"""Domain types for the v0 episode runtime.

Episodes consist of ordered ``Tick`` records (one per logged phase). Intents represent
the planner output before policy and act phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class EpisodeStatus(str, Enum):
    """Lifecycle state of an episode."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    POLICY_BLOCKED = "policy_blocked"


class TickPhase(str, Enum):
    """Phase within one orchestrator cycle (PRAL-style pipeline)."""

    PERCEPTION = "perception"
    REASON = "reason"
    POLICY = "policy"
    ACT = "act"
    LEARN = "learn"


@dataclass
class Observation:
    """Structured outcome from the environment or a tool invocation.

    Attributes:
        ok: Whether the observation represents success for downstream logic.
        payload: Arbitrary structured detail (errors, echoed values, final text).
    """

    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolIntent:
    """Planner decision to invoke a named tool with arguments.

    Attributes:
        tool: Registered tool name.
        args: Arguments passed to the tool implementation.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalIntent:
    """Planner decision to finish the episode with a user-visible message.

    Attributes:
        message: Terminal artifact text.
    """

    message: str


Intent = ToolIntent | FinalIntent


@dataclass(frozen=True)
class PolicyResult:
    """Policy gate outcome before executing an intent.

    Attributes:
        allowed: Whether execution may proceed.
        reason: Machine-oriented explanation when ``allowed`` is False.
    """

    allowed: bool
    reason: str = ""


@dataclass
class Tick:
    """Single logged phase within an episode.

    Attributes:
        index: Monotonic index across the whole episode (append order).
        cycle: Orchestrator cycle counter (replan increments).
        phase: Which pipeline phase produced this tick.
        data: Serializable payload for tracing (intent snapshot, flags, etc.).
    """

    index: int
    cycle: int
    phase: TickPhase
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Mutable aggregate for one episode run.

    Attributes:
        id: Stable identifier (e.g. UUID string).
        task: Original task text from the trigger.
        status: Current lifecycle state.
        ticks: Ordered trace of phase records.
        max_cycles: Upper bound on orchestrator cycles (replan budget).
        artifact: Terminal message or failure token when finished.
    """

    id: str
    task: str
    status: EpisodeStatus
    ticks: list[Tick] = field(default_factory=list)
    max_cycles: int = 16
    artifact: str | None = None

    @staticmethod
    def new(task: str, max_cycles: int = 16) -> Episode:
        """Create a new running episode with a fresh id.

        Args:
            task: User task text (may be empty).
            max_cycles: Maximum orchestrator cycles before forced failure.

        Returns:
            Episode instance in ``RUNNING`` state with no ticks yet.
        """
        return Episode(
            id=str(uuid4()),
            task=task,
            status=EpisodeStatus.RUNNING,
            max_cycles=max_cycles,
        )

    def append_tick(self, cycle: int, phase: TickPhase, **data: Any) -> Tick:
        """Append one tick and return it.

        Args:
            cycle: Current orchestrator cycle index.
            phase: Phase being logged.
            **data: Extra structured fields stored under ``Tick.data``.

        Returns:
            The newly appended ``Tick``.
        """
        idx = len(self.ticks)
        tick = Tick(index=idx, cycle=cycle, phase=phase, data=dict(data))
        self.ticks.append(tick)
        return tick
