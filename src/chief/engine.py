"""Episode engine implementing Perception → Reason → Policy → Act → Learn.

The outer loop advances orchestrator cycles on replan (failed tool observation).
"""

from __future__ import annotations

from typing import Mapping

from chief.brain import Brain
from chief.domain import (
    Episode,
    EpisodeStatus,
    FinalIntent,
    Intent,
    Observation,
    TickPhase,
    ToolIntent,
)
from chief.memory import MemorySession, append_episode_jsonl
from chief.policy import evaluate_intent
from chief.config import RuntimeConfig
from chief.tools import ToolFn


def _intent_to_dict(intent: Intent) -> dict:
    """Serialize an intent for tick logging.

    Args:
        intent: Planner output.

    Returns:
        JSON-friendly dict tagged with ``type`` discriminator.
    """
    if isinstance(intent, ToolIntent):
        return {"type": "tool", "tool": intent.tool, "args": intent.args}
    return {"type": "final", "message": intent.message}


async def run_episode(
    task: str,
    *,
    runtime: RuntimeConfig,
    brain: Brain,
    tools: Mapping[str, ToolFn],
    max_cycles: int | None = None,
) -> Episode:
    """Execute one episode until completion, policy block, or cycle budget exhaustion.

    Each loop iteration logs perception, reason, policy, act, and learn phases.
    Failed tool observations increment the cycle counter so the planner can replan.

    Args:
        task: Task text passed to the planner and echoed tools.
        runtime: Process configuration snapshot (episode limits, policy allowlist, …).
        brain: Planner implementation (:class:`~chief.brain.Brain`).
        tools: Registry mapping tool names to callables returning ``Observation``.
        max_cycles: Maximum orchestrator cycles; ``None`` uses ``runtime.episode_max_cycles``.

    Returns:
        Episode aggregate with terminal ``status`` and optional ``artifact`` message.
    """
    mc = max_cycles if max_cycles is not None else runtime.episode_max_cycles
    allowed = runtime.allowed_tools_policy
    episode = Episode.new(task, max_cycles=mc)
    memory = MemorySession()
    cycle = 0

    def log(phase: TickPhase, **data: object) -> None:
        """Append one tick to the episode and append the same record to JSONL.

        Args:
            phase: Pipeline phase for this tick.
            **data: Arbitrary structured fields stored on the tick (JSON-serializable).
        """
        t = episode.append_tick(cycle, phase, **{k: v for k, v in data.items()})
        append_episode_jsonl(episode.id, t)

    while episode.status == EpisodeStatus.RUNNING:
        if cycle >= episode.max_cycles:
            episode.status = EpisodeStatus.FAILED
            episode.artifact = "max_cycles_exceeded"
            break

        log(
            TickPhase.PERCEPTION,
            has_prior_obs=memory.last_observation() is not None,
            last_ok=memory.last_observation().ok if memory.last_observation() else None,
        )

        try:
            intent = await brain.reason(memory, task)
        except Exception as exc:
            log(TickPhase.REASON, intent=None, error=str(exc))
            episode.status = EpisodeStatus.FAILED
            episode.artifact = f"brain_error:{exc!r}"
            break
        log(TickPhase.REASON, intent=_intent_to_dict(intent))

        policy = evaluate_intent(intent, cycle, episode.max_cycles, allowed_tools=allowed)
        log(TickPhase.POLICY, allowed=policy.allowed, reason=policy.reason)
        if not policy.allowed:
            episode.status = EpisodeStatus.POLICY_BLOCKED
            episode.artifact = policy.reason
            break

        if isinstance(intent, FinalIntent):
            obs = Observation(ok=True, payload={"final": intent.message})
            log(TickPhase.ACT, kind="final", message=intent.message)
        else:
            fn = tools.get(intent.tool)
            if fn is None:
                obs = Observation(
                    ok=False,
                    payload={"error": f"unknown_tool:{intent.tool}"},
                )
            else:
                obs = fn(intent.args)
            log(TickPhase.ACT, tool=intent.tool, args=intent.args, ok=obs.ok)

        memory.observations.append(obs)
        memory.record("observation", {"ok": obs.ok, "payload": obs.payload})
        log(TickPhase.LEARN, observation_ok=obs.ok, payload=obs.payload)

        if isinstance(intent, FinalIntent):
            episode.status = EpisodeStatus.COMPLETED
            episode.artifact = intent.message
            break

        if obs.ok:
            cycle += 1
            continue

        cycle += 1

    if episode.status == EpisodeStatus.RUNNING:
        episode.status = EpisodeStatus.FAILED
        episode.artifact = episode.artifact or "engine_exited_unexpectedly"

    return episode
