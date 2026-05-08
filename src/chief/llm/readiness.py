"""LLM readiness states and assessment (Phase 2).

Maps registry rows to :class:`LlmReadiness` and runs :mod:`chief.llm.probe` when configuration
looks actionable. ``fake`` is always :data:`LlmReadiness.READY` without network I/O.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Final

import platformdirs

from chief.config.runtime import RuntimeConfig
from chief.config.providers_registry import ProviderRecord
from chief.llm.probe import probe_provider_record

_PLACEHOLDER_API_KEY: Final = "REPLACE_ME_OR_USE_ENV"
_ENV_SKIP_PROBE: Final = "CHIEF_SKIP_LLM_PROBE"


class LlmReadiness(str, Enum):
    """High-level LLM availability for the running process."""

    READY = "ready"
    BLOCKED = "blocked"
    UNCONFIGURED = "unconfigured"


class LlmNotReadyError(RuntimeError):
    """Raised when an LLM planner is requested but readiness is not :data:`LlmReadiness.READY`.

    Attributes:
        state: Readiness enum value.
        detail: Human-readable explanation (safe for stderr; no secrets).
    """

    def __init__(self, state: LlmReadiness, detail: str) -> None:
        """Initialize with readiness state and user-facing detail text.

        Args:
            state: Outcome of :func:`assess_provider_id` or :func:`assess_provider_readiness`.
            detail: Safe explanation without secret material.
        """
        self.state = state
        self.detail = detail
        super().__init__(detail)


def user_providers_hint() -> str:
    """Return the default user overlay path for provider JSON (for error messages).

    Returns:
        Absolute path hint to ``providers.json`` under XDG config.
    """
    base = Path(platformdirs.user_config_dir("chief", appauthor=False)).expanduser()
    return str(base / "providers.json")


def skip_llm_probe_from_env() -> bool:
    """Return true if probes should be skipped (tests, air-gapped scripts).

    Honors ``CHIEF_SKIP_LLM_PROBE`` set to ``1``, ``true``, or ``yes`` (case-insensitive).
    """
    raw = os.environ.get(_ENV_SKIP_PROBE, "").strip().lower()
    return raw in ("1", "true", "yes")


def is_placeholder_api_key(value: str | None) -> bool:
    """Return true if the key is missing or still a bundled placeholder string."""
    if value is None or not str(value).strip():
        return True
    return str(value).strip() == _PLACEHOLDER_API_KEY


def provider_configuration_issue(record: ProviderRecord) -> str | None:
    """Return a short reason if the row should be treated as UNCONFIGURED, else None.

    Args:
        record: Merged registry row.

    Returns:
        Non-empty message when the provider must not be probed as-is, or ``None`` if probing is OK.
    """
    if not record.enabled:
        return (
            f"provider {record.id!r} is disabled (enabled=false). "
            f"Edit registry: {user_providers_hint()}"
        )
    if record.kind == "openai" and is_placeholder_api_key(record.api_key):
        return (
            "openai provider needs a real API key (not a placeholder). "
            f"Edit registry or env: {user_providers_hint()}"
        )
    if record.kind == "anthropic" and is_placeholder_api_key(record.api_key):
        return (
            "anthropic provider needs a real API key. "
            f"Edit registry: {user_providers_hint()}"
        )
    if record.kind == "gemini" and is_placeholder_api_key(record.api_key):
        return (
            "gemini provider needs a real API key. "
            f"Edit registry: {user_providers_hint()}"
        )
    return None


async def assess_provider_readiness(
    record: ProviderRecord,
    runtime: RuntimeConfig,
) -> tuple[LlmReadiness, str]:
    """Classify one registry row after optional probe.

    Args:
        record: Provider to assess (typically enabled).
        runtime: Full snapshot (used to align OpenAI-wire probes with the model catalog).

    Returns:
        ``(state, message)`` where ``message`` is empty on :data:`LlmReadiness.READY`.

    Note:
        Does not raise on transport errors; failed probes yield :data:`LlmReadiness.BLOCKED`.
    """
    issue = provider_configuration_issue(record)
    if issue is not None:
        return LlmReadiness.UNCONFIGURED, issue
    if skip_llm_probe_from_env():
        return LlmReadiness.READY, ""
    ok = await probe_provider_record(record, runtime=runtime)
    if ok:
        return LlmReadiness.READY, ""
    return (
        LlmReadiness.BLOCKED,
        (
            f"probe failed for provider {record.id!r} ({record.kind}). "
            "Check base URL, model id, network, and API key. "
            f"Registry file: {user_providers_hint()}"
        ),
    )


async def assess_provider_id(runtime: RuntimeConfig, provider_id: str) -> tuple[LlmReadiness, str]:
    """Assess readiness for a CLI / IPC planner id (``fake`` or registry ``id``).

    Args:
        runtime: Built runtime snapshot.
        provider_id: Raw id (case-insensitive for ``fake`` and registry keys).

    Returns:
        ``(state, message)`` suitable for logging or :class:`LlmNotReadyError`.
    """
    key = provider_id.strip().lower()
    if key == "fake":
        return LlmReadiness.READY, ""
    rec = runtime.providers_by_id.get(key)
    if rec is None:
        return (
            LlmReadiness.UNCONFIGURED,
            f"unknown provider {provider_id!r}. Known: {', '.join(sorted(runtime.providers_by_id))}",
        )
    return await assess_provider_readiness(rec, runtime)


async def ensure_llm_ready_or_raise(runtime: RuntimeConfig, provider_id: str) -> None:
    """Raise :class:`LlmNotReadyError` unless the provider is ready for LLM planning.

    Args:
        runtime: Process configuration.
        provider_id: Planner id for the upcoming episode.

    Raises:
        LlmNotReadyError: When state is not :data:`LlmReadiness.READY`.
    """
    state, msg = await assess_provider_id(runtime, provider_id)
    if state is not LlmReadiness.READY:
        raise LlmNotReadyError(state, msg)
