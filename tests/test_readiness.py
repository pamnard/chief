"""Tests for LLM readiness classification and probe wiring."""

from __future__ import annotations

import dataclasses

import pytest

from chief.brain_select import select_brain
from chief.config import build_runtime_config
from chief.config.providers_registry import ProviderRecord
from chief.llm import probe
from chief.llm.readiness import (
    LlmReadiness,
    assess_provider_id,
    assess_provider_readiness,
    provider_configuration_issue,
    skip_llm_probe_from_env,
)


def test_provider_configuration_issue_disabled() -> None:
    """Disabled rows are UNCONFIGURED with a hint."""
    r = ProviderRecord(
        id="x",
        kind="custom_llm",
        base_url="http://127.0.0.1:1/v1",
        model="m",
        api_key=None,
        timeout_seconds=1.0,
        api_version=None,
        max_tokens=None,
        enabled=False,
    )
    msg = provider_configuration_issue(r)
    assert msg is not None
    assert "disabled" in msg.lower()


def test_provider_configuration_issue_openai_placeholder() -> None:
    """Vendor OpenAI with placeholder key is not probe-ready."""
    r = ProviderRecord(
        id="openai",
        kind="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="REPLACE_ME_OR_USE_ENV",
        timeout_seconds=30.0,
        api_version=None,
        max_tokens=None,
        enabled=True,
    )
    msg = provider_configuration_issue(r)
    assert msg is not None
    assert "api key" in msg.lower()


@pytest.mark.asyncio
async def test_assess_disabled_provider() -> None:
    """Disabled canonical id yields UNCONFIGURED without calling the network."""
    rt = build_runtime_config()
    c = rt.providers_by_id["custom_llm"]
    off = dataclasses.replace(c, enabled=False)
    rt2 = dataclasses.replace(
        rt,
        providers_by_id={**rt.providers_by_id, "custom_llm": off},
    )
    st, msg = await assess_provider_id(rt2, "custom_llm")
    assert st is LlmReadiness.UNCONFIGURED
    assert "disabled" in msg.lower()


@pytest.mark.asyncio
async def test_assess_fake_skips_probe() -> None:
    """Planner fake is always READY."""
    rt = build_runtime_config()
    st, msg = await assess_provider_id(rt, "fake")
    assert st is LlmReadiness.READY
    assert msg == ""


def test_select_brain_disabled_raises() -> None:
    """Registry rows with enabled=false cannot be selected."""
    rt = build_runtime_config()
    c = rt.providers_by_id["custom_llm"]
    off = dataclasses.replace(c, enabled=False)
    rt2 = dataclasses.replace(
        rt,
        providers_by_id={**rt.providers_by_id, "custom_llm": off},
    )
    with pytest.raises(ValueError, match="disabled"):
        select_brain("custom_llm", rt2)


@pytest.mark.asyncio
async def test_probe_openai_wire_models_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /models 200 short-circuits Chat Completions probe."""

    class _Resp:
        status_code = 200

        @property
        def is_success(self) -> bool:
            return True

    class _Client:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def get(self, *a: object, **kw: object) -> _Resp:
            return _Resp()

        async def post(self, *a: object, **kw: object) -> _Resp:
            raise AssertionError("POST should not run when GET /models succeeds")

    monkeypatch.setattr(probe.httpx, "AsyncClient", _Client)
    rec = ProviderRecord(
        id="custom_llm",
        kind="custom_llm",
        base_url="http://127.0.0.1:11434/v1",
        model="m",
        api_key=None,
        timeout_seconds=30.0,
        api_version=None,
        max_tokens=None,
        enabled=True,
    )
    assert await probe.probe_provider_record(rec) is True


@pytest.mark.asyncio
async def test_skip_probe_env_marks_ready_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CHIEF_SKIP_LLM_PROBE=1 skips HTTP and returns READY if config is actionable."""
    monkeypatch.setenv("CHIEF_SKIP_LLM_PROBE", "1")
    assert skip_llm_probe_from_env() is True
    rt = build_runtime_config()
    rec = rt.providers_by_id["custom_llm"]
    st, msg = await assess_provider_readiness(rec, rt)
    assert st is LlmReadiness.READY
    assert msg == ""
