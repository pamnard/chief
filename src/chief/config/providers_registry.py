"""Load and validate the provider registry from bundled JSON and optional user overlay."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import platformdirs

_KIND_CUSTOM_LLM: Final = "custom_llm"
_KIND_OPENAI: Final = "openai"
_KIND_ANTHROPIC: Final = "anthropic"
_KIND_GEMINI: Final = "gemini"

_OPENAI_VENDOR_DEFAULT_BASE: Final = "https://api.openai.com/v1"

KNOWN_KINDS: Final[frozenset[str]] = frozenset(
    {_KIND_CUSTOM_LLM, _KIND_OPENAI, _KIND_ANTHROPIC, _KIND_GEMINI}
)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class ProviderRecord:
    """One provider instance from the registry (JSON object).

    Attributes:
        id: Stable key used by CLI and IPC (lowercase ``a-z0-9_-``).
        kind: ``custom_llm`` (OpenAI Chat Completions wire, arbitrary base URL), ``openai``
            (vendor OpenAI deployment), ``anthropic``, or ``gemini``.
        base_url: API root; for ``openai`` may be omitted to use the official host default.
        model: Default model id for this provider.
        api_key: Optional bearer / key material; empty strings become ``None`` where allowed.
        timeout_seconds: HTTP client timeout for one request.
        api_version: ``anthropic`` only: ``anthropic-version`` header.
        max_tokens: ``anthropic`` only: generation cap.
        enabled: When false, the instance is ignored by probes and cannot be selected as a planner.
    """

    id: str
    kind: str
    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float
    api_version: str | None
    max_tokens: int | None
    enabled: bool


def _defaults_providers_path() -> Path:
    """Return the path to packaged ``defaults.providers.json``."""
    return Path(__file__).resolve().parent / "defaults.providers.json"


def _xdg_providers_path() -> Path:
    """Return the default user overlay path under XDG config."""
    base = platformdirs.user_config_dir("chief", appauthor=False)
    return Path(base) / "providers.json"


def _read_providers_file(path: Path) -> list[dict[str, Any]]:
    """Parse a registry file; return the ``providers`` list as dict rows.

    Raises:
        FileNotFoundError: If ``path`` is missing.
        ValueError: If JSON is invalid or not an object with a ``providers`` array.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        root: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(root, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    items = root.get("providers")
    if not isinstance(items, list):
        raise ValueError(f"{path}: missing providers array")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: providers[{i}] must be object")
        out.append(item)
    return out


def _parse_bool(val: Any, *, path: Path, index: int, field: str) -> bool:
    """Coerce a JSON value to bool."""
    if isinstance(val, bool):
        return val
    raise ValueError(f"{path}: providers[{index}].{field} must be boolean")


def _parse_float(val: Any, *, path: Path, index: int, field: str) -> float:
    """Coerce a JSON value to float."""
    if isinstance(val, bool):
        raise ValueError(f"{path}: providers[{index}].{field} must be number")
    if isinstance(val, int | float):
        return float(val)
    if isinstance(val, str) and val.strip():
        return float(val.strip())
    raise ValueError(f"{path}: providers[{index}].{field} must be number")


def _parse_int(val: Any, *, path: Path, index: int, field: str) -> int:
    """Coerce a JSON value to int."""
    if isinstance(val, bool):
        raise ValueError(f"{path}: providers[{index}].{field} must be int")
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str) and val.strip():
        return int(val.strip())
    raise ValueError(f"{path}: providers[{index}].{field} must be int")


def _parse_record(row: dict[str, Any], *, path: Path, index: int) -> ProviderRecord:
    """Validate and normalize one element of the ``providers`` array.

    Applies kind-specific rules (required fields, default OpenAI base URL, Anthropic headers).

    Args:
        row: Raw object from JSON.
        path: Source file path (for error messages).
        index: Zero-based index inside ``providers``.

    Returns:
        Frozen :class:`ProviderRecord`.

    Raises:
        ValueError: On invalid id, unknown kind, missing required fields, or type errors.
    """
    pid = str(row.get("id", "")).strip()
    if not pid or not _ID_PATTERN.match(pid):
        raise ValueError(
            f"{path}: providers[{index}].id must match {_ID_PATTERN.pattern!r}, got {pid!r}"
        )
    kind = str(row.get("kind", "")).strip()
    if kind not in KNOWN_KINDS:
        raise ValueError(f"{path}: providers[{index}].kind must be one of {sorted(KNOWN_KINDS)!r}")

    base_url = str(row.get("base_url", "")).strip()
    model = str(row.get("model", "")).strip()
    timeout_seconds = _parse_float(row.get("timeout_seconds", 120.0), path=path, index=index, field="timeout_seconds")

    api_raw = row.get("api_key", None)
    if api_raw is None:
        api_key: str | None = None
    else:
        s = str(api_raw).strip()
        api_key = s if s else None

    api_version_raw = row.get("api_version", None)
    api_version = str(api_version_raw).strip() if api_version_raw is not None else None
    if api_version == "":
        api_version = None

    max_tokens: int | None = None
    if "max_tokens" in row and row["max_tokens"] is not None:
        max_tokens = _parse_int(row["max_tokens"], path=path, index=index, field="max_tokens")

    enabled = True
    if "enabled" in row and row["enabled"] is not None:
        enabled = _parse_bool(row["enabled"], path=path, index=index, field="enabled")

    if kind == _KIND_CUSTOM_LLM:
        if not base_url:
            raise ValueError(f"{path}: providers[{index}] custom_llm requires base_url")
        if not model:
            raise ValueError(f"{path}: providers[{index}] custom_llm requires model")
        return ProviderRecord(
            id=pid,
            kind=kind,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            api_version=None,
            max_tokens=None,
            enabled=enabled,
        )

    if kind == _KIND_OPENAI:
        resolved_base = base_url if base_url else _OPENAI_VENDOR_DEFAULT_BASE
        if not model:
            raise ValueError(f"{path}: providers[{index}] openai requires model")
        return ProviderRecord(
            id=pid,
            kind=kind,
            base_url=resolved_base,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            api_version=None,
            max_tokens=None,
            enabled=enabled,
        )

    if kind == _KIND_ANTHROPIC:
        if not base_url:
            raise ValueError(f"{path}: providers[{index}] anthropic requires base_url")
        if not api_version:
            raise ValueError(f"{path}: providers[{index}] anthropic requires api_version")
        if not api_key:
            raise ValueError(f"{path}: providers[{index}] anthropic requires api_key")
        if not model:
            raise ValueError(f"{path}: providers[{index}] anthropic requires model")
        if max_tokens is None:
            raise ValueError(f"{path}: providers[{index}] anthropic requires max_tokens")
        return ProviderRecord(
            id=pid,
            kind=kind,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            api_version=api_version,
            max_tokens=max_tokens,
            enabled=enabled,
        )

    # gemini
    if not base_url:
        raise ValueError(f"{path}: providers[{index}] gemini requires base_url")
    if not api_key:
        raise ValueError(f"{path}: providers[{index}] gemini requires api_key")
    if not model:
        raise ValueError(f"{path}: providers[{index}] gemini requires model")
    return ProviderRecord(
        id=pid,
        kind=kind,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        api_version=None,
        max_tokens=None,
        enabled=enabled,
    )


def _load_file_as_map(path: Path) -> dict[str, ProviderRecord]:
    """Load a registry file and return id → record."""
    rows = _read_providers_file(path)
    by_id: dict[str, ProviderRecord] = {}
    for i, row in enumerate(rows):
        rec = _parse_record(row, path=path, index=i)
        if rec.id in by_id:
            raise ValueError(f"{path}: duplicate provider id {rec.id!r}")
        by_id[rec.id] = rec
    return by_id


def load_merged_providers() -> tuple[ProviderRecord, ...]:
    """Load bundled defaults, then merge user overlay (by ``id``).

    Merge order: packaged ``defaults.providers.json``; then if ``CHIEF_PROVIDERS_FILE`` is set,
    that file (must exist); else if ``$XDG_CONFIG_HOME/chief/providers.json`` exists, that file.
    Later sources **replace** entries with the same ``id`` and may add new ids.

    Returns:
        Non-empty tuple of :class:`ProviderRecord`, sorted by ``id``.

    Raises:
        ValueError: If validation fails or the merged registry is empty.
    """
    bundled_path = _defaults_providers_path()
    if not bundled_path.is_file():
        raise ValueError(f"missing bundled providers registry: {bundled_path}")

    merged = _load_file_as_map(bundled_path)

    overlay_raw = os.environ.get("CHIEF_PROVIDERS_FILE", "").strip()
    if overlay_raw:
        overlay_path = Path(overlay_raw).expanduser().resolve()
        if not overlay_path.is_file():
            raise ValueError(f"CHIEF_PROVIDERS_FILE points to missing file: {overlay_path}")
        overlay = _load_file_as_map(overlay_path)
        merged.update(overlay)
    else:
        user_path = _xdg_providers_path()
        if user_path.is_file():
            merged.update(_load_file_as_map(user_path))

    if not merged:
        raise ValueError("provider registry is empty after merge")

    return tuple(sorted(merged.values(), key=lambda r: r.id))
