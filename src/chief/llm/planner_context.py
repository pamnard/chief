"""Shared planner text for all HTTP brains in :mod:`chief.llm.providers`.

Vendor-neutral: same system instructions and episode serialization for OpenAI-compatible Chat Completions,
Anthropic Messages, Gemini ``generateContent``, etc. Wire-specific JSON is built in ``chief.llm.schema.*``;
transport + endpoint + model live in each ``Http*Brain`` class. System prompt text lives on
:class:`~chief.config.runtime.RuntimeConfig` (built once per process).
"""

from __future__ import annotations

import json

from chief.memory import MemorySession


def serialize_episode_context(memory: MemorySession, task: str) -> str:
    """Build user-side planner text listing the task and prior observations.

    Args:
        memory: Session containing observations from earlier act phases in this episode.
        task: Original task string from the trigger.

    Returns:
        Multi-line string suitable as the user message alongside a fixed system prompt.
    """
    lines = [f"Task: {task}"]
    for i, obs in enumerate(memory.observations, start=1):
        payload = json.dumps(obs.payload, ensure_ascii=False)
        lines.append(f"Observation {i}: ok={obs.ok} payload={payload}")
    return "\n".join(lines)
