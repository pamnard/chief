"""Shared pytest fixtures for the chief test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Use deterministic ``fake`` planner unless the environment already overrides it.

    Packaged defaults target ``custom_llm`` for real installs; the suite must stay network-free
    without requiring every test to monkeypatch ``CHIEF_DEFAULT_PROVIDER``.
    """
    os.environ.setdefault("CHIEF_DEFAULT_PROVIDER", "fake")


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
