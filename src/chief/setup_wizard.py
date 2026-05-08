"""Interactive bootstrap for XDG provider files (Phase 2 minimal wizard)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import platformdirs
import tomllib
import tomli_w


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a same-directory temp file and replace.

    Args:
        path: Destination file (parent dirs created as needed).
        text: UTF-8 content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _prompt(label: str, default: str) -> str:
    """Read one line; blank input selects ``default``.

    Args:
        label: Prompt text shown to the user.
        default: Value used when input is empty.

    Returns:
        Stripped non-empty string.
    """
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def merge_chief_preferences(
    chief_toml: Path,
    *,
    default_provider: str,
    default_model: str | None,
) -> None:
    """Merge ``[chief]`` keys into ``chief_toml`` and write atomically.

    Preserves other top-level tables. Does not remove ``default_model`` when
    ``default_model`` is ``None``.

    Args:
        chief_toml: Path to ``chief.toml`` under XDG config.
        default_provider: ``[chief].default_provider`` value.
        default_model: Optional ``[chief].default_model`` catalog id.

    Raises:
        OSError: On filesystem errors.
        ValueError: If existing TOML cannot be parsed.
    """
    data: dict[str, Any] = {}
    if chief_toml.is_file():
        data = tomllib.loads(chief_toml.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{chief_toml}: root must be a table")
    chief_sec = data.get("chief")
    if not isinstance(chief_sec, dict):
        chief_sec = {}
    chief_sec["default_provider"] = default_provider
    if default_model:
        chief_sec["default_model"] = default_model
    data["chief"] = chief_sec
    _atomic_write_text(chief_toml, tomli_w.dumps(data))


def upsert_custom_llm_provider(
    path: Path,
    *,
    base_url: str,
    model: str,
    timeout_seconds: float = 120.0,
) -> None:
    """Merge or create ``providers.json`` with canonical ``custom_llm`` row.

    Args:
        path: ``providers.json`` path.
        base_url: Chat Completions API root (``.../v1``).
        model: Default model id for requests.
        timeout_seconds: Per-request timeout.

    Raises:
        OSError: On filesystem errors.
        ValueError: If existing JSON is not an object with a ``providers`` list.
    """
    if path.is_file():
        root: Any = json.loads(path.read_text(encoding="utf-8"))
    else:
        root = {"providers": []}
    if not isinstance(root, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    providers = root.get("providers")
    if not isinstance(providers, list):
        raise ValueError(f"{path}: missing providers array")
    row = {
        "id": "custom_llm",
        "kind": "custom_llm",
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_key": "",
        "timeout_seconds": timeout_seconds,
        "enabled": True,
    }
    by_id: dict[str, dict[str, Any]] = {}
    for item in providers:
        if isinstance(item, dict) and str(item.get("id", "")).strip():
            by_id[str(item["id"]).strip()] = item
    by_id["custom_llm"] = row
    root["providers"] = sorted(by_id.values(), key=lambda x: str(x.get("id", "")))
    text = json.dumps(root, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(path, text)


def run_setup_providers() -> int:
    """Run interactive prompts and write ``chief.toml`` + ``providers.json``.

    Returns:
        ``0`` on success, ``1`` on I/O or parse error, ``2`` when not a TTY.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "chief setup providers: requires an interactive terminal (TTY).",
            file=sys.stderr,
        )
        print(
            "Edit chief.toml and providers.json under your XDG config directory, "
            "or set CHIEF_DEFAULT_PROVIDER / CHIEF_PROVIDERS_FILE.",
            file=sys.stderr,
        )
        return 2

    print("chief — provider bootstrap (writes XDG config)", flush=True)
    url_default = "http://127.0.0.1:11434/v1"
    model_default = "gemma4:e4b"
    base_url = _prompt("OpenAI-compat base URL (e.g. Ollama …/v1)", url_default)
    model = _prompt("Default API model id", model_default)
    catalog_id = _prompt("Optional catalog default_model id (blank to skip)", "").strip() or None

    cfg_root = Path(platformdirs.user_config_dir("chief", appauthor=False))
    chief_toml = cfg_root / "chief.toml"
    prov_json = cfg_root / "providers.json"
    try:
        upsert_custom_llm_provider(prov_json, base_url=base_url, model=model)
        merge_chief_preferences(
            chief_toml,
            default_provider="custom_llm",
            default_model=catalog_id,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"chief setup providers: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {chief_toml}", flush=True)
    print(f"Wrote {prov_json}", flush=True)
    if catalog_id:
        print(
            f"Set [chief].default_model = {catalog_id!r} "
            "(must exist in the merged model catalog).",
            flush=True,
        )
    print('Try: chief run "quick test"', flush=True)
    return 0
