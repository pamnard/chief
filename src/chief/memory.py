"""Session memory and append-only episodic logging.

``MemorySession`` holds short-term observations for the active episode.
``append_episode_jsonl`` persists each tick as one JSON line on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chief.domain import Observation, Tick
from chief.paths import episodes_log_dir


@dataclass
class MemorySession:
    """In-memory scratchpad for one episode.

    Observations accumulate so the planner can replan after tool failures.

    Attributes:
        observations: Ordered tool/environment observations.
        events: Lightweight structured log lines for debugging extensions.
    """

    observations: list[Observation] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        """Append a labeled event for diagnostics.

        Args:
            kind: Short category label (e.g. ``\"observation\"``).
            payload: Arbitrary JSON-serializable fields.
        """
        self.events.append({"kind": kind, **payload})

    def last_observation(self) -> Observation | None:
        """Return the latest observation, if any.

        Returns:
            Last ``Observation`` or ``None`` when empty.
        """
        return self.observations[-1] if self.observations else None


def append_episode_jsonl(episode_id: str, tick: Tick) -> Path:
    """Append one JSON line describing ``tick`` to the episode's JSONL file.

    Args:
        episode_id: Episode identifier (UUID string).
        tick: Phase record to serialize.

    Returns:
        Path to the JSONL file written or appended.
    """
    path = episodes_log_dir() / f"{episode_id}.jsonl"
    line = json.dumps(
        {
            "index": tick.index,
            "cycle": tick.cycle,
            "phase": tick.phase.value,
            "data": tick.data,
        },
        ensure_ascii=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
