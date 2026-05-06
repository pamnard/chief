"""Merge bundled defaults → optional user files → ``CHIEF_*`` env.

Precedence (last wins for overlapping keys): **defaults.toml** (package) →
**XDG** ``$XDG_CONFIG_HOME/chief/chief.toml`` (or ``~/.config/chief/chief.toml``) →
**``CHIEF_CONFIG``** path if set → **env** for mapped keys.
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

import platformdirs

_ENV_OVERRIDES: list[tuple[str, str, str, str]] = [
    ("custom_llm", "base_url", "CHIEF_LLM_BASE_URL", "str"),
    ("custom_llm", "model", "CHIEF_LLM_MODEL", "str"),
    ("custom_llm", "api_key", "CHIEF_LLM_API_KEY", "str"),
    ("custom_llm", "timeout_seconds", "CHIEF_LLM_TIMEOUT", "float"),
    ("custom_llm", "json_mode", "CHIEF_LLM_JSON_MODE", "bool"),
    ("openai", "vendor_api_base", "CHIEF_OPENAI_BASE_URL", "str"),
    ("openai", "model", "CHIEF_OPENAI_MODEL", "str"),
    ("openai", "api_key", "CHIEF_OPENAI_API_KEY", "str"),
    ("openai", "timeout_seconds", "CHIEF_OPENAI_TIMEOUT", "float"),
    ("openai", "json_mode", "CHIEF_OPENAI_JSON_MODE", "bool"),
    ("anthropic", "vendor_api_base", "CHIEF_ANTHROPIC_BASE_URL", "str"),
    ("anthropic", "api_version", "CHIEF_ANTHROPIC_API_VERSION", "str"),
    ("anthropic", "api_key", "CHIEF_ANTHROPIC_API_KEY", "str"),
    ("anthropic", "model", "CHIEF_ANTHROPIC_MODEL", "str"),
    ("anthropic", "timeout_seconds", "CHIEF_ANTHROPIC_TIMEOUT", "float"),
    ("anthropic", "max_tokens", "CHIEF_ANTHROPIC_MAX_TOKENS", "int"),
    ("gemini", "vendor_api_base", "CHIEF_GEMINI_BASE_URL", "str"),
    ("gemini", "api_key", "CHIEF_GEMINI_API_KEY", "str"),
    ("gemini", "model", "CHIEF_GEMINI_MODEL", "str"),
    ("gemini", "timeout_seconds", "CHIEF_GEMINI_TIMEOUT", "float"),
]


def _defaults_path() -> Path:
    return Path(__file__).resolve().parent / "defaults.toml"


def _xdg_user_path() -> Path:
    base = platformdirs.user_config_dir("chief", appauthor=False)
    return Path(base) / "chief.toml"


def _merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_env_value(raw: str, typ: str) -> Any:
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
    """Full merged configuration (mutate only on your own copy if needed)."""
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
    block = cfg.get(name)
    return block if isinstance(block, dict) else {}


def str_from(cfg: dict[str, Any], table: str, key: str) -> str:
    """Read non-empty string from merged config (env already merged in :func:`load_merged_config`)."""
    sec = section(cfg, table)
    if key not in sec:
        raise KeyError(f"missing [{table}].{key} in merged config")
    v = sec[key]
    s = str(v).strip() if v is not None else ""
    if not s:
        raise ValueError(f"[{table}].{key} is empty in merged config")
    return s


def optional_str_from(cfg: dict[str, Any], table: str, key: str) -> str | None:
    sec = section(cfg, table)
    if key not in sec:
        return None
    v = sec[key]
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def float_from(cfg: dict[str, Any], table: str, key: str) -> float:
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
    sec = section(cfg, table)
    if key not in sec:
        raise KeyError(f"missing [{table}].{key} in merged config")
    v = sec[key]
    if isinstance(v, bool):
        return v
    raise TypeError(f"[{table}].{key} must be boolean in merged config")


def llm_integration_enabled(cfg: dict[str, Any] | None = None) -> bool:
    c = cfg if cfg is not None else load_merged_config()
    sec = section(c, "test")
    if "enable_llm_integration" not in sec:
        return False
    v = sec["enable_llm_integration"]
    if not isinstance(v, bool):
        raise ValueError("[test].enable_llm_integration must be boolean in merged config")
    return v


def planner_allowed_tools(cfg: dict[str, Any] | None = None) -> tuple[str, ...]:
    c = cfg if cfg is not None else load_merged_config()
    sec = section(c, "planner")
    raw = sec.get("allowed_tools")
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError("[planner].allowed_tools must be a list of strings in merged config")
    return tuple(str(x) for x in raw)


def episode_max_cycles(cfg: dict[str, Any] | None = None) -> int:
    c = cfg if cfg is not None else load_merged_config()
    v = section(c, "episode").get("max_cycles")
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError("[episode].max_cycles must be int in merged config")
    return int(v)
