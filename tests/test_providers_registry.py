"""Unit tests for :mod:`chief.config.providers_registry`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chief.config.providers_registry import ProviderRecord, load_merged_providers


def test_provider_record_openai_optional_key() -> None:
    """Empty api_key string becomes None for custom_llm."""
    r = ProviderRecord(
        id="x",
        kind="custom_llm",
        base_url="http://127.0.0.1:11434/v1",
        model="m",
        api_key=None,
        timeout_seconds=30.0,
        api_version=None,
        max_tokens=None,
        enabled=True,
    )
    assert r.api_key is None


def test_load_merged_providers_non_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundled defaults must yield at least one provider."""
    monkeypatch.delenv("CHIEF_PROVIDERS_FILE", raising=False)
    prov = load_merged_providers()
    assert len(prov) >= 1
    ids = {p.id for p in prov}
    assert "custom_llm" in ids
    assert all(p.kind in ("custom_llm", "openai", "anthropic", "gemini") for p in prov)


def test_chief_providers_file_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CHIEF_PROVIDERS_FILE replaces entries with same id and can add new ones."""
    overlay = {
        "providers": [
            {
                "id": "custom_llm",
                "kind": "custom_llm",
                "base_url": "http://example.invalid:9/v1",
                "model": "overlay-model",
                "api_key": "",
                "timeout_seconds": 1.0,
            },
            {
                "id": "extra",
                "kind": "custom_llm",
                "base_url": "http://127.0.0.1:2/v1",
                "model": "m2",
                "api_key": "",
                "timeout_seconds": 5.0,
            },
        ]
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(overlay), encoding="utf-8")
    monkeypatch.setenv("CHIEF_PROVIDERS_FILE", str(p))
    prov = load_merged_providers()
    by_id = {x.id: x for x in prov}
    assert by_id["custom_llm"].model == "overlay-model"
    assert by_id["custom_llm"].base_url == "http://example.invalid:9/v1"
    assert "extra" in by_id
