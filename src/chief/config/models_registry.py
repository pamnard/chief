"""Load and validate the **model catalog** (Phase 3): user-declared models and metadata.

Bundled ``defaults.models.json`` may be empty. User overlay at
``$XDG_CONFIG_HOME/chief/models.json`` merges by ``id`` (same rules as the provider registry).
Override path: ``CHIEF_MODELS_FILE``.

Each row names a **logical** model id, the **provider** it belongs to, the **API model** string,
and optional flags used when building requests (e.g. OpenAI Chat Completions ``json_object`` mode).
Optional ``technical`` marks models suitable for **auxiliary** work (context compression, self-check,
routing metadata) versus primary user-facing planning — similar to a separate "fast" model in other agents.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

import platformdirs

from chief.config.providers_registry import _ID_PATTERN

_CONTEXT_MIN: Final = 1
_CONTEXT_MAX: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """One catalog entry: which provider and which API model id to use.

    Attributes:
        id: Stable key for ``[chief].default_model`` and future routing (``a-z0-9_-``).
        provider_id: Must match an ``id`` in the merged **provider** registry.
        model: Model string sent to the vendor API (e.g. OpenAI ``model`` field).
        json_mode: For ``custom_llm`` / ``openai`` Chat Completions only: request JSON-shaped output.
        context_tokens: Optional advertised context window (tokens); reserved for routing / UI.
        supports_tools: Whether the model is expected to support tool-style planner fields; reserved.
        technical: When ``True``, the row may be selected for non-user-facing auxiliary LLM calls
            (e.g. summarization, validation). Primary chat / planner defaults typically use rows
            with ``technical=False``.
    """

    id: str
    provider_id: str
    model: str
    json_mode: bool
    context_tokens: int | None
    supports_tools: bool
    technical: bool


def _defaults_models_path() -> Path:
    """Return the path to packaged ``defaults.models.json``."""
    return Path(__file__).resolve().parent / "defaults.models.json"


def _xdg_models_path() -> Path:
    """Return the default user overlay path under XDG config."""
    base = platformdirs.user_config_dir("chief", appauthor=False)
    return Path(base) / "models.json"


def _read_models_file(path: Path) -> list[dict[str, Any]]:
    """Parse a model catalog file; return the ``models`` list as dict rows.

    Raises:
        FileNotFoundError: If ``path`` is missing.
        ValueError: If JSON is invalid or not an object with a ``models`` array.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        root: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(root, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    items = root.get("models")
    if not isinstance(items, list):
        raise ValueError(f"{path}: missing models array")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: models[{i}] must be object")
        out.append(item)
    return out


def _parse_bool(val: Any, *, path: Path, index: int, field: str) -> bool:
    """Coerce a JSON value to bool."""
    if isinstance(val, bool):
        return val
    raise ValueError(f"{path}: models[{index}].{field} must be boolean")


def _parse_optional_int(val: Any, *, path: Path, index: int, field: str) -> int | None:
    """Parse optional positive int or JSON null.

    Raises:
        ValueError: If the value is not an int, null, or stringified int in range.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        raise ValueError(f"{path}: models[{index}].{field} must be int or null")
    if isinstance(val, int):
        n = int(val)
    elif isinstance(val, float):
        n = int(val)
    elif isinstance(val, str) and val.strip():
        n = int(val.strip())
    else:
        raise ValueError(f"{path}: models[{index}].{field} must be int or null")
    if n < _CONTEXT_MIN or n > _CONTEXT_MAX:
        raise ValueError(
            f"{path}: models[{index}].{field} must be between {_CONTEXT_MIN} and {_CONTEXT_MAX}"
        )
    return n


def _parse_record(row: dict[str, Any], *, path: Path, index: int) -> ModelRecord:
    """Validate and normalize one element of the ``models`` array.

    Args:
        row: Raw object from JSON.
        path: Source file path (for error messages).
        index: Zero-based index inside ``models``.

    Returns:
        Frozen :class:`ModelRecord`.

    Raises:
        ValueError: On invalid id, missing ``model``, bad types, or out-of-range ``context_tokens``.
    """
    mid = str(row.get("id", "")).strip()
    if not mid or not _ID_PATTERN.match(mid):
        raise ValueError(
            f"{path}: models[{index}].id must match {_ID_PATTERN.pattern!r}, got {mid!r}"
        )
    pid = str(row.get("provider_id", "")).strip()
    if not pid or not _ID_PATTERN.match(pid):
        raise ValueError(f"{path}: models[{index}].provider_id must be a valid id, got {pid!r}")
    model = str(row.get("model", "")).strip()
    if not model:
        raise ValueError(f"{path}: models[{index}].model is required")

    json_mode = _parse_bool(row.get("json_mode", False), path=path, index=index, field="json_mode")
    supports_tools = _parse_bool(
        row.get("supports_tools", False), path=path, index=index, field="supports_tools"
    )
    technical = _parse_bool(row.get("technical", False), path=path, index=index, field="technical")
    context_tokens: int | None = None
    if "context_tokens" in row:
        context_tokens = _parse_optional_int(
            row.get("context_tokens"), path=path, index=index, field="context_tokens"
        )

    return ModelRecord(
        id=mid,
        provider_id=pid,
        model=model,
        json_mode=json_mode,
        context_tokens=context_tokens,
        supports_tools=supports_tools,
        technical=technical,
    )


def _load_file_as_map(path: Path) -> dict[str, ModelRecord]:
    """Load a catalog file and return id → record.

    Args:
        path: JSON file containing a top-level ``models`` array.

    Returns:
        Mapping validated with :func:`_parse_record`.

    Raises:
        ValueError: On duplicate ``id`` or parse errors.
    """
    rows = _read_models_file(path)
    by_id: dict[str, ModelRecord] = {}
    for i, row in enumerate(rows):
        rec = _parse_record(row, path=path, index=i)
        if rec.id in by_id:
            raise ValueError(f"{path}: duplicate model id {rec.id!r}")
        by_id[rec.id] = rec
    return by_id


def load_merged_models() -> tuple[ModelRecord, ...]:
    """Load bundled defaults, then merge user overlay (by ``id``).

    Merge order: packaged ``defaults.models.json``; then if ``CHIEF_MODELS_FILE`` is set, that
    file (must exist); else if ``$XDG_CONFIG_HOME/chief/models.json`` exists, that file. Later
    sources replace entries with the same ``id`` and may add new ids.

    Returns:
        Tuple of :class:`ModelRecord`, sorted by ``id``.

    Raises:
        ValueError: If validation fails.
    """
    bundled_path = _defaults_models_path()
    if not bundled_path.is_file():
        raise ValueError(f"missing bundled models catalog: {bundled_path}")

    merged = _load_file_as_map(bundled_path)

    overlay_raw = os.environ.get("CHIEF_MODELS_FILE", "").strip()
    if overlay_raw:
        overlay_path = Path(overlay_raw).expanduser().resolve()
        if not overlay_path.is_file():
            raise ValueError(f"CHIEF_MODELS_FILE points to missing file: {overlay_path}")
        merged.update(_load_file_as_map(overlay_path))
    else:
        user_path = _xdg_models_path()
        if user_path.is_file():
            merged.update(_load_file_as_map(user_path))

    return tuple(sorted(merged.values(), key=lambda r: r.id))


def technical_model_candidates(models: Iterable[ModelRecord]) -> tuple[ModelRecord, ...]:
    """Return catalog rows marked for auxiliary (non-primary) LLM use.

    Args:
        models: Iterable of merged :class:`ModelRecord` instances (e.g. from
            :func:`load_merged_models` or ``runtime.models``).

    Returns:
        Tuple of records with ``technical is True``, sorted by ``id``.
    """
    return tuple(sorted((m for m in models if m.technical), key=lambda r: r.id))
