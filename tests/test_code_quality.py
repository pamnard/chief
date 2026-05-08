"""Meta-tests: docstring style and coverage (Google convention, interrogate).

Requires optional dev dependencies ``pydocstyle`` and ``interrogate`` (``pip install -e ".[dev]"``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHIEF_SRC = _REPO_ROOT / "src" / "chief"
_TESTS_DIR = _REPO_ROOT / "tests"


def test_pydocstyle_google_convention() -> None:
    """Fail if ``src/chief`` or ``tests`` violate Google-style docstrings (pydocstyle)."""
    pytest.importorskip("pydocstyle")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pydocstyle",
            str(_CHIEF_SRC),
            str(_TESTS_DIR),
            "--convention=google",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stdout or "") + (proc.stderr or "")
        pytest.fail(f"pydocstyle failed (exit {proc.returncode}):\n{msg or '(no output)'}")


def test_interrogate_docstring_coverage() -> None:
    """Fail if docstring coverage under ``src/chief`` drops below 100%."""
    pytest.importorskip("interrogate")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "interrogate",
            str(_CHIEF_SRC),
            "--quiet",
            "--fail-under",
            "100",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stdout or "") + (proc.stderr or "")
        pytest.fail(f"interrogate failed (exit {proc.returncode}):\n{msg or '(no output)'}")
