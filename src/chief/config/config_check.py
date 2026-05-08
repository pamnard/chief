"""Static (non-network) checks for LLM-related configuration before requests."""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

from chief.config.runtime import RuntimeConfig

_SETUP_HINT = (
    "Create or edit files under your XDG config directory for chief "
    "(typically ~/.config/chief/: chief.toml, providers.json, models.json), "
    "or set CHIEF_DEFAULT_PROVIDER / CHIEF_PROVIDERS_FILE / CHIEF_MODELS_FILE. "
    "Interactive TTY: chief setup providers. "
    "See README for keys and registry shape."
)


def setup_hint_message() -> str:
    """Return a short hint pointing users at local configuration files.

    Returns:
        Single-line message suitable for stderr.
    """
    return _SETUP_HINT


def user_llm_overlay_present() -> bool:
    """Return True if the user supplied any chief-specific config outside bundled defaults.

    Used for level-1 presence: when the effective planner is not ``fake`` but nothing in
    XDG/env points at user intent, we warn (bundled defaults alone are not considered
    "your" LLM setup).

    Returns:
        ``True`` when a user chief.toml, extra TOML path, provider/model overlay path, or
        ``CHIEF_DEFAULT_*`` env override is present.
    """
    cfg_dir = Path(platformdirs.user_config_dir("chief", appauthor=False))
    if (cfg_dir / "chief.toml").is_file():
        return True
    if (cfg_dir / "providers.json").is_file():
        return True
    if (cfg_dir / "models.json").is_file():
        return True
    for env_key in ("CHIEF_CONFIG", "CHIEF_CONFIG_FILE"):
        raw = os.environ.get(env_key, "").strip()
        if raw and Path(raw).expanduser().resolve().is_file():
            return True
    if os.environ.get("CHIEF_DEFAULT_PROVIDER", "").strip():
        return True
    if os.environ.get("CHIEF_DEFAULT_MODEL", "").strip():
        return True
    prov = os.environ.get("CHIEF_PROVIDERS_FILE", "").strip()
    if prov and Path(prov).expanduser().resolve().is_file():
        return True
    models = os.environ.get("CHIEF_MODELS_FILE", "").strip()
    if models and Path(models).expanduser().resolve().is_file():
        return True
    return False


def static_llm_config_issues(
    rt: RuntimeConfig,
    *,
    effective_provider_id: str,
) -> tuple[str, ...]:
    """Return human-readable static configuration errors for a chosen planner id.

    Does not perform HTTP probes. When the tuple is non-empty, the caller should treat
    configuration as invalid and abort before LLM calls.

    Args:
        rt: Frozen runtime from :func:`~chief.config.runtime.build_runtime_config`.
        effective_provider_id: Planner id (``fake`` or a registry id), e.g. CLI default.

    Returns:
        Zero or more error lines (no trailing newlines).
    """
    pid = effective_provider_id.strip().lower()
    if pid == "fake":
        return ()

    rec = rt.providers_by_id.get(pid)
    if rec is None:
        return (f"unknown provider {effective_provider_id!r} (not in registry)",)

    if not rec.enabled:
        return (f"provider {pid!r} is disabled in the registry (enabled=false)",)

    mid = rt.default_model_id
    if mid:
        spec = rt.models_by_id.get(mid)
        if spec is not None and spec.provider_id != pid:
            return (
                f"[chief].default_model={mid!r} is bound to provider {spec.provider_id!r}, "
                f"but the active planner is {pid!r} (change default_model, default_provider, "
                f"or pass --provider)",
            )

    key = rec.api_key or ""
    if key and "REPLACE_ME" in key.upper():
        return (
            f"provider {pid!r} uses a placeholder API key; replace it in providers.json "
            f"(or your CHIEF_PROVIDERS_FILE overlay)",
        )

    return ()
