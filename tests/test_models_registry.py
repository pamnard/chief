"""Tests for :mod:`chief.config.models_registry` and catalog resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chief.config import (
    build_runtime_config,
    load_merged_models,
    openai_wire_model_and_json_mode,
    technical_model_candidates,
)


def test_load_merged_models_non_empty_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundled defaults.models.json must load (may be an empty models array)."""
    monkeypatch.delenv("CHIEF_MODELS_FILE", raising=False)
    models = load_merged_models()
    assert isinstance(models, tuple)


def test_chief_models_file_overlay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CHIEF_MODELS_FILE replaces/adds catalog entries."""
    overlay = {
        "models": [
            {
                "id": "local-fast",
                "provider_id": "custom_llm",
                "model": "gemma2:2b",
                "json_mode": True,
                "context_tokens": 8192,
                "supports_tools": False,
            }
        ]
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(overlay), encoding="utf-8")
    monkeypatch.setenv("CHIEF_MODELS_FILE", str(p))
    models = load_merged_models()
    by_id = {m.id: m for m in models}
    assert by_id["local-fast"].model == "gemma2:2b"
    assert by_id["local-fast"].json_mode is True
    assert by_id["local-fast"].context_tokens == 8192
    assert by_id["local-fast"].technical is False


def test_model_catalog_technical_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``technical`` defaults false; ``technical_model_candidates`` filters true rows."""
    overlay = {
        "models": [
            {
                "id": "main-m",
                "provider_id": "custom_llm",
                "model": "big",
                "json_mode": False,
                "supports_tools": False,
            },
            {
                "id": "aux-m",
                "provider_id": "custom_llm",
                "model": "small",
                "json_mode": False,
                "supports_tools": False,
                "technical": True,
            },
        ]
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(overlay), encoding="utf-8")
    monkeypatch.setenv("CHIEF_MODELS_FILE", str(p))

    models = load_merged_models()
    by_id = {m.id: m for m in models}
    assert by_id["main-m"].technical is False
    assert by_id["aux-m"].technical is True
    tech_ids = {x.id for x in technical_model_candidates(models)}
    assert "aux-m" in tech_ids
    assert "main-m" not in tech_ids


def test_default_model_resolves_openai_wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """[chief].default_model + catalog row overrides provider registry model and json_mode."""
    overlay = {
        "models": [
            {
                "id": "m1",
                "provider_id": "custom_llm",
                "model": "from-catalog",
                "json_mode": True,
            }
        ]
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(overlay), encoding="utf-8")
    monkeypatch.setenv("CHIEF_MODELS_FILE", str(p))
    monkeypatch.setenv("CHIEF_DEFAULT_MODEL", "m1")
    rt = build_runtime_config()
    assert rt.default_model_id == "m1"
    rec = rt.providers_by_id["custom_llm"]
    mid, jm = openai_wire_model_and_json_mode(rt, rec)
    assert mid == "from-catalog"
    assert jm is True


def test_unknown_default_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_runtime_config`` fails when ``CHIEF_DEFAULT_MODEL`` is not in the catalog."""
    monkeypatch.setenv("CHIEF_DEFAULT_MODEL", "nope")
    with pytest.raises(ValueError, match="default_model"):
        build_runtime_config()
