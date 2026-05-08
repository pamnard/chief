"""Tests for static LLM configuration checks (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chief.config import build_runtime_config
from chief.config.config_check import (
    static_llm_config_issues,
    user_llm_overlay_present,
)
def test_static_llm_config_issues_fake_empty() -> None:
    """``fake`` planner skips static LLM errors."""
    rt = build_runtime_config()
    assert static_llm_config_issues(rt, effective_provider_id="fake") == ()


def test_static_llm_config_issues_unknown_provider() -> None:
    """Unknown registry id yields an error line."""
    rt = build_runtime_config()
    out = static_llm_config_issues(rt, effective_provider_id="no_such_provider")
    assert len(out) == 1
    assert "unknown provider" in out[0]


def test_static_llm_config_issues_placeholder_api_key() -> None:
    """Vendor rows with REPLACE_ME in api_key are rejected."""
    rt = build_runtime_config()
    out = static_llm_config_issues(rt, effective_provider_id="openai")
    assert out
    assert "placeholder" in out[0].lower()


def test_static_llm_config_issues_model_provider_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``default_model`` must reference the effective provider when set."""
    models = {
        "models": [
            {
                "id": "m1",
                "provider_id": "custom_llm",
                "model": "x",
                "json_mode": False,
                "supports_tools": False,
            }
        ]
    }
    p = tmp_path / "models.json"
    p.write_text(json.dumps(models), encoding="utf-8")
    monkeypatch.setenv("CHIEF_MODELS_FILE", str(p))
    monkeypatch.setenv("CHIEF_DEFAULT_MODEL", "m1")
    rt = build_runtime_config()
    out = static_llm_config_issues(rt, effective_provider_id="openai")
    assert out
    assert "default_model" in out[0]


def test_build_runtime_rejects_disabled_default_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``[chief].default_provider`` must not point at an ``enabled=false`` row."""
    monkeypatch.delenv("CHIEF_DEFAULT_PROVIDER", raising=False)
    cfg = tmp_path / "chief.toml"
    cfg.write_text('[chief]\ndefault_provider = "custom_llm"\n', encoding="utf-8")
    monkeypatch.setenv("CHIEF_CONFIG", str(cfg))
    overlay = {
        "providers": [
            {
                "id": "custom_llm",
                "kind": "custom_llm",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "gemma4:e4b",
                "api_key": "",
                "timeout_seconds": 120,
                "enabled": False,
            }
        ]
    }
    pj = tmp_path / "providers.json"
    pj.write_text(json.dumps(overlay), encoding="utf-8")
    monkeypatch.setenv("CHIEF_PROVIDERS_FILE", str(pj))
    with pytest.raises(ValueError, match="disabled"):
        build_runtime_config()


def test_cli_run_fails_on_placeholder_openai(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """``chief run`` with OpenAI and bundled placeholder key exits before the episode."""
    from chief.cli import main

    monkeypatch.setenv("CHIEF_DEFAULT_PROVIDER", "openai")
    code = main(["run", "noop"])
    assert code == 2
    err = capsys.readouterr().err
    assert "placeholder" in err.lower()


def test_user_llm_overlay_env_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """``CHIEF_DEFAULT_PROVIDER`` counts as user overlay."""
    monkeypatch.setenv("CHIEF_DEFAULT_PROVIDER", "custom_llm")
    assert user_llm_overlay_present() is True
    monkeypatch.delenv("CHIEF_DEFAULT_PROVIDER", raising=False)
