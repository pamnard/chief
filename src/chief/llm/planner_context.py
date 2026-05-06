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
    """Build user-side text listing task and prior observations."""
    lines = [f"Task: {task}"]
    for i, obs in enumerate(memory.observations, start=1):
        payload = json.dumps(obs.payload, ensure_ascii=False)
        lines.append(f"Observation {i}: ok={obs.ok} payload={payload}")
    return "\n".join(lines)
