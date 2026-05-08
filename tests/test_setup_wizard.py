"""Tests for :mod:`chief.setup_wizard` (non-network)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from chief.setup_wizard import merge_chief_preferences, run_setup_providers, upsert_custom_llm_provider


def test_merge_chief_preserves_other_tables(tmp_path: Path) -> None:
    """Merging ``[chief]`` leaves other top-level tables intact."""
    p = tmp_path / "chief.toml"
    p.write_text(
        "[planner]\nallowed_tools = [\"noop\"]\n\n[chief]\ndefault_provider = \"fake\"\n",
        encoding="utf-8",
    )
    merge_chief_preferences(p, default_provider="custom_llm", default_model=None)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert data["chief"]["default_provider"] == "custom_llm"
    assert data["planner"]["allowed_tools"] == ["noop"]


def test_upsert_custom_llm_merges_ids(tmp_path: Path) -> None:
    """``providers.json`` keeps unrelated provider rows."""
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "id": "local_chat_api",
                        "kind": "custom_llm",
                        "base_url": "http://127.0.0.1:8001/api/v1",
                        "model": "m",
                        "api_key": "",
                        "timeout_seconds": 60,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    upsert_custom_llm_provider(path, base_url="http://h/v1", model="x")
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = {r["id"] for r in data["providers"]}
    assert ids == {"custom_llm", "local_chat_api"}
    by = {r["id"]: r for r in data["providers"]}
    assert by["custom_llm"]["model"] == "x"


def test_run_setup_providers_requires_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a TTY the wizard refuses to run."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert run_setup_providers() == 2


def test_run_setup_providers_writes_xdg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Happy path writes chief.toml and providers.json under a fake XDG root."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(
        "chief.setup_wizard.platformdirs.user_config_dir",
        lambda app, appauthor=False: str(tmp_path),
    )
    inputs = iter(["", "", ""])  # all defaults; optional catalog blank
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert run_setup_providers() == 0
    chief = tomllib.loads((tmp_path / "chief.toml").read_text(encoding="utf-8"))
    assert chief["chief"]["default_provider"] == "custom_llm"
    prov = json.loads((tmp_path / "providers.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in prov["providers"]}
    assert by_id["custom_llm"]["model"] == "gemma4:e4b"
    assert "11434" in by_id["custom_llm"]["base_url"]
