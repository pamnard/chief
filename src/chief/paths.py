"""Filesystem paths aligned with XDG Base Directory via ``platformdirs``.

State data (episode traces) lives under the user state directory by default.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_state_dir


def episodes_log_dir() -> Path:
    """Return (and create) the directory that stores per-episode JSONL traces.

    Returns:
        Absolute path to ``.../chief/episodes`` under ``user_state_dir``.
    """
    base = Path(user_state_dir("chief", appauthor=False))
    d = base / "episodes"
    d.mkdir(parents=True, exist_ok=True)
    return d
