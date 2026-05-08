"""Merge bundled defaults → optional user files → ``CHIEF_*`` env.

Precedence (last wins for overlapping keys): **defaults.toml** (package) →
**XDG** ``$XDG_CONFIG_HOME/chief/chief.toml`` (or ``~/.config/chief/chief.toml``) →
**``CHIEF_CONFIG``** path if set → **env** for mapped keys.

LLM endpoints live in the **provider registry** (``defaults.providers.json`` + optional
``providers.json``); see :mod:`chief.config.providers_registry`. The **model catalog**
(``defaults.models.json`` + optional ``models.json``) is separate; see
:mod:`chief.config.models_registry`.
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

import platformdirs

_ENV_OVERRIDES: list[tuple[str, str, str, str]] = [
    ("chief", "default_provider", "CHIEF_DEFAULT_PROVIDER", "str"),
    ("chief", "default_model", "CHIEF_DEFAULT_MODEL", "str"),
    ("serve", "socket_path", "CHIEF_SERVE_SOCKET", "str"),
]


def _defaults_path() -> Path:
    """Return the path to bundled ``defaults.toml``."""
    return Path(__file__).resolve().parent / "defaults.toml"


def _xdg_user_path() -> Path:
    """Return the path to the optional user override file under XDG config."""
    base = platformdirs.user_config_dir("chief", appauthor=False)
    return Path(base) / "chief.toml"


def _merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overlay`` into a copy of ``base`` (nested dicts merged recursively)."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_env_value(raw: str, typ: str) -> Any:
    """Parse a non-empty environment string into a scalar of the given type label.

    Raises:
        ValueError: If ``typ`` is unknown or a boolean string is not parseable.
    """
    raw = raw.strip()
    if typ == "str":
        return raw
    if typ == "float":
        return float(raw)
    if typ == "int":
        return int(raw)
    if typ == "bool":
        low = raw.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"invalid boolean env value {raw!r}")
    raise ValueError(f"unknown type {typ!r}")


def _apply_env(merged: dict[str, Any]) -> None:
    """Apply ``_ENV_OVERRIDES`` and integration-test flags onto ``merged`` in place.

    Raises:
        ValueError: If a non-string override env var is set but empty, or if a boolean
            env value cannot be parsed (via :func:`_parse_env_value`).
    """
    for tbl, key, env_key, typ in _ENV_OVERRIDES:
        if env_key not in os.environ:
            continue
        raw = os.environ.get(env_key, "")
        if typ == "str" and not raw.strip():
            continue
        if typ != "str" and not str(raw).strip():
            raise ValueError(f"env {env_key} is set but empty")
        val = _parse_env_value(raw, typ)
        merged.setdefault(tbl, {})
        if not isinstance(merged[tbl], dict):
            merged[tbl] = {}
        merged[tbl][key] = val

    if os.environ.get("CHIEF_TEST_LLM", "").strip() == "1":
        merged.setdefault("test", {})
        merged["test"]["enable_llm_integration"] = True


def load_merged_config() -> dict[str, Any]:
    """Load and merge all configuration sources into one dict.

    Merge order: packaged defaults, XDG user file, ``CHIEF_CONFIG`` / ``CHIEF_CONFIG_FILE``,
    then environment overrides.

    Returns:
        Merged top-level mapping (same object should be treated as read-only by callers
        unless they intentionally copy). For a stable typed snapshot per process, prefer
        :func:`chief.config.runtime.build_runtime_config` instead of calling this repeatedly.
    """
    with _defaults_path().open("rb") as f:
        merged: dict[str, Any] = tomllib.load(f)

    xdg = _xdg_user_path()
    if xdg.is_file():
        with xdg.open("rb") as f:
            user = tomllib.load(f)
        if isinstance(user, dict):
            merged = _merge_dict(merged, user)

    for env_key in ("CHIEF_CONFIG", "CHIEF_CONFIG_FILE"):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            with p.open("rb") as f:
                extra = tomllib.load(f)
            if isinstance(extra, dict):
                merged = _merge_dict(merged, extra)

    _apply_env(merged)
    return merged


def section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    r"""Return a top-level TOML table as a dict, or an empty dict if missing or not a mapping.

    Args:
        cfg: Merged configuration from :func:`load_merged_config`.
        name: Top-level table name (e.g. ``"planner"``).

    Returns:
        Table contents, or ``{}`` if absent or not a dict.
    """
    block = cfg.get(name)
    return block if isinstance(block, dict) else {}


def str_from(cfg: dict[str, Any], table: str, key: str) -> str:
    """Read a required non-empty string from a merged-config table.

    Args:
        cfg: Merged configuration (environment already applied by :func:`load_merged_config`).
        table: Top-level table name.
        key: Key within that table.

    Returns:
        Stripped non-empty string.

    Raises:
        KeyError: If the key is missing.
        ValueError: If the value is empty after stripping.
    """
    sec = section(cfg, table)
    if key not in sec:
        raise KeyError(f"missing [{table}].{key} in merged config")
    v = sec[key]
    s = str(v).strip() if v is not None else ""
    if not s:
        raise ValueError(f"[{table}].{key} is empty in merged config")
    return s


def optional_str_from(cfg: dict[str, Any], table: str, key: str) -> str | None:
    """Read an optional string; missing key or blank value becomes ``None``.

    Args:
        cfg: Merged configuration.
        table: Top-level table name.
        key: Key within that table.

    Returns:
        Stripped string, or ``None`` if absent, null, or blank.
    """
    sec = section(cfg, table)
    if key not in sec:
        return None
    v = sec[key]
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def float_from(cfg: dict[str, Any], table: str, key: str) -> float:
    """Read a required floating-point value from a merged-config table.

    Args:
        cfg: Merged configuration.
        table: Top-level table name.
        key: Key within that table.

    Returns:
        Value as ``float``.

    Raises:
        KeyError: If the key is missing.
        TypeError: If the value cannot be interpreted as a number (including booleans).
    """
    sec = section(cfg, table)
    if key not in sec:
        raise KeyError(f"missing [{table}].{key} in merged config")
    v = sec[key]
    if isinstance(v, bool):
        raise TypeError(f"[{table}].{key} must be number")
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str) and v.strip():
        return float(v.strip())
    raise TypeError(f"[{table}].{key} must be number in merged config")


def int_from(cfg: dict[str, Any], table: str, key: str) -> int:
    """Read a required integer from a merged-config table.

    Args:
        cfg: Merged configuration.
        table: Top-level table name.
        key: Key within that table.

    Returns:
        Value as ``int``.

    Raises:
        KeyError: If the key is missing.
        TypeError: If the value cannot be interpreted as an int (including booleans).
    """
    sec = section(cfg, table)
    if key not in sec:
        raise KeyError(f"missing [{table}].{key} in merged config")
    v = sec[key]
    if isinstance(v, bool):
        raise TypeError(f"[{table}].{key} must be int")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str) and v.strip():
        return int(v.strip())
    raise TypeError(f"[{table}].{key} must be int in merged config")


def bool_from(cfg: dict[str, Any], table: str, key: str) -> bool:
    """Read a required boolean from a merged-config table.

    Args:
        cfg: Merged configuration.
        table: Top-level table name.
        key: Key within that table.

    Returns:
        Boolean value.

    Raises:
        KeyError: If the key is missing.
        TypeError: If the value is not a JSON/TOML boolean.
    """
    sec = section(cfg, table)
    if key not in sec:
        raise KeyError(f"missing [{table}].{key} in merged config")
    v = sec[key]
    if isinstance(v, bool):
        return v
    raise TypeError(f"[{table}].{key} must be boolean in merged config")


def llm_integration_enabled(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether optional live-LLM integration tests are enabled.

    Args:
        cfg: Merged configuration, or ``None`` to call :func:`load_merged_config` once.

    Returns:
        ``True`` if ``[test].enable_llm_integration`` is boolean ``True``.

    Raises:
        ValueError: If the key exists but is not a boolean.
    """
    c = cfg if cfg is not None else load_merged_config()
    sec = section(c, "test")
    if "enable_llm_integration" not in sec:
        return False
    v = sec["enable_llm_integration"]
    if not isinstance(v, bool):
        raise ValueError("[test].enable_llm_integration must be boolean in merged config")
    return v


def planner_allowed_tools(cfg: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return the ordered allowlist of tool names from ``[planner].allowed_tools``.

    Args:
        cfg: Merged configuration, or ``None`` to call :func:`load_merged_config` once.

    Returns:
        Tuple of tool name strings in list order.

    Raises:
        ValueError: If ``allowed_tools`` is missing or not a list of strings.
    """
    c = cfg if cfg is not None else load_merged_config()
    sec = section(c, "planner")
    raw = sec.get("allowed_tools")
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError("[planner].allowed_tools must be a list of strings in merged config")
    return tuple(str(x) for x in raw)


def episode_max_cycles(cfg: dict[str, Any] | None = None) -> int:
    """Return ``[episode].max_cycles`` from merged configuration.

    Args:
        cfg: Merged configuration, or ``None`` to call :func:`load_merged_config` once.

    Returns:
        Positive cycle budget as int (caller validates range if needed).

    Raises:
        TypeError: If the value is not an ``int`` (including bool).
    """
    c = cfg if cfg is not None else load_merged_config()
    v = section(c, "episode").get("max_cycles")
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError("[episode].max_cycles must be int in merged config")
    return int(v)
