"""Unit tests for the episode engine (no network).

Uses ``FakeBrain`` and filesystem isolation for JSONL logs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chief.brain import Brain, FakeBrain
from chief.domain import EpisodeStatus, ToolIntent
from chief.engine import run_episode
from chief.memory import MemorySession
from chief.config import RuntimeConfig, build_runtime_config
from chief.tools import build_registry


@pytest.fixture
def runtime() -> RuntimeConfig:
    """Single config snapshot per test (one merge from disk / env)."""
    return build_runtime_config()


class BrokenBrain:
    """Brain implementation that always requests the ``broken`` tool."""

    async def reason(self, memory: MemorySession, task: str) -> ToolIntent:
        """Return a failing tool intent on every call.

        Args:
            memory: Unused; kept for protocol compatibility.
            task: Unused.

        Returns:
            ``ToolIntent`` targeting ``broken``.
        """
        return ToolIntent("broken", {})


@pytest.fixture
def isolated_episode_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect episode JSONL output into a temporary directory.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Path to the fake episodes root directory.
    """
    log_root = tmp_path / "episodes"
    log_root.mkdir(parents=True, exist_ok=True)

    def _dir() -> Path:
        return log_root

    monkeypatch.setattr("chief.memory.episodes_log_dir", _dir)
    return log_root


async def test_fake_brain_completes_with_replan(
    isolated_episode_logs: Path, runtime: RuntimeConfig
) -> None:
    """FakeBrain should recover after a simulated failure and emit a final artifact."""
    ep = await run_episode(
        "hello",
        runtime=runtime,
        brain=FakeBrain(runtime),
        tools=build_registry(runtime),
        max_cycles=8,
    )
    assert ep.status == EpisodeStatus.COMPLETED
    assert ep.artifact and "hello" in ep.artifact
    phases = [t.phase.value for t in ep.ticks]
    assert phases.count("reason") >= 2
    log_file = isolated_episode_logs / f"{ep.id}.jsonl"
    assert log_file.is_file()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(ep.ticks)


async def test_max_cycles_exceeded(isolated_episode_logs: Path, runtime: RuntimeConfig) -> None:
    """Budget exhaustion should fail the episode before completion."""
    ep = await run_episode(
        "x",
        runtime=runtime,
        brain=BrokenBrain(),
        tools=build_registry(runtime),
        max_cycles=2,
    )
    assert ep.status == EpisodeStatus.FAILED
    assert ep.artifact == "max_cycles_exceeded"


async def test_brain_protocol_fake(runtime: RuntimeConfig) -> None:
    """Runtime-checkable protocol should accept FakeBrain."""
    b: Brain = FakeBrain(runtime)
    assert isinstance(b, Brain)


class RaisingBrain:
    """Brain that always raises, to exercise engine error handling."""

    async def reason(self, memory: MemorySession, task: str) -> ToolIntent:
        """Always raise for deterministic failure tests.

        Args:
            memory: Unused.
            task: Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("boom")


async def test_brain_exception_marks_episode_failed(
    isolated_episode_logs: Path, runtime: RuntimeConfig
) -> None:
    """Planner exceptions should surface as ``FAILED`` with ``brain_error`` prefix."""
    ep = await run_episode(
        "x",
        runtime=runtime,
        brain=RaisingBrain(),
        tools=build_registry(runtime),
        max_cycles=4,
    )
    assert ep.status == EpisodeStatus.FAILED
    assert ep.artifact is not None
    assert "brain_error" in ep.artifact
    assert any(t.phase.value == "reason" and "error" in t.data for t in ep.ticks)
