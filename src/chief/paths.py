"""Filesystem paths aligned with XDG Base Directory via ``platformdirs``.

State data (episode traces) lives under the user state directory by default.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_runtime_dir, user_state_dir


def episodes_log_dir() -> Path:
    """Return (and create) the directory that stores per-episode JSONL traces.

    Returns:
        Absolute path to ``.../chief/episodes`` under ``user_state_dir``.
    """
    base = Path(user_state_dir("chief", appauthor=False))
    d = base / "episodes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_serve_socket_path() -> Path:
    """Return the default Unix socket path for ``chief serve`` under XDG runtime.

    The parent directory is created if missing (same pattern as episode logs).

    Returns:
        Absolute path ``…/chief.sock`` under :func:`platformdirs.user_runtime_dir` for ``chief``.
    """
    base = Path(user_runtime_dir("chief", appauthor=False))
    base.mkdir(parents=True, exist_ok=True)
    return base / "chief.sock"
