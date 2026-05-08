"""Typed process-wide configuration: merged TOML, provider registry JSON, and derived slices.

``build_runtime_config`` reads the filesystem once and returns an immutable :class:`RuntimeConfig`
snapshot. Canonical registry ids ``custom_llm``, ``openai``, ``anthropic``, and ``gemini`` populate
the same :class:`CustomLlmSlice` / :class:`OpenAiSlice` / … fields that historically came from TOML
tables with those names, so :meth:`CustomChatCompletionsBrain.from_runtime` and siblings stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chief.config.loader import (
    episode_max_cycles,
    load_merged_config,
    llm_integration_enabled as _llm_integration_enabled_from_mapping,
    optional_str_from,
    planner_allowed_tools,
    str_from,
)
from chief.config.models_registry import ModelRecord, load_merged_models
from chief.config.providers_registry import ProviderRecord, load_merged_providers
from chief.paths import default_serve_socket_path

_CANONICAL_CUSTOM_LLM = "custom_llm"
_CANONICAL_OPENAI = "openai"
_CANONICAL_ANTHROPIC = "anthropic"
_CANONICAL_GEMINI = "gemini"


@dataclass(frozen=True, slots=True)
class CustomLlmSlice:
    """OpenAI Chat Completions wire settings for a **custom** base URL (registry ``custom_llm``).

    Populated from the provider registry row with id ``custom_llm`` and kind ``custom_llm``.

    Attributes:
        base_url: HTTP root ending in ``/v1`` (trailing slashes stripped by callers as needed).
        model: Default model id for requests.
        api_key: Optional bearer token; may be ``None`` for local gateways.
        timeout_seconds: Client timeout for a single completion request.
    """

    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class OpenAiSlice:
    """Vendor OpenAI Chat Completions settings (registry ``openai``).

    ``vendor_api_base`` is the resolved API root (official default applied when the JSON row omits
    ``base_url``).

    Attributes:
        vendor_api_base: Resolved ``https://api.openai.com/v1``-style base URL.
        model: Default OpenAI model id.
        api_key: API key string (may be empty if unset in config; vendor calls typically require it).
        timeout_seconds: Client timeout for one request.
    """

    vendor_api_base: str
    model: str
    api_key: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class AnthropicSlice:
    """Anthropic Messages API settings (registry ``anthropic``).

    Attributes:
        vendor_api_base: Messages API base URL (typically ``https://api.anthropic.com/v1``).
        api_version: ``anthropic-version`` header value.
        api_key: API key for ``x-api-key``.
        model: Default model id.
        timeout_seconds: Client timeout for one request.
        max_tokens: Maximum tokens to generate per call.
    """

    vendor_api_base: str
    api_version: str
    api_key: str
    model: str
    timeout_seconds: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class GeminiSlice:
    """Google Generative Language ``generateContent`` settings (registry ``gemini``).

    Attributes:
        vendor_api_base: API root (e.g. ``v1beta`` base URL).
        api_key: Key appended as ``?key=`` on requests.
        model: Default Gemini model id.
        timeout_seconds: Client timeout for one request.
    """

    vendor_api_base: str
    api_key: str
    model: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Immutable snapshot of merged TOML plus provider registry used at process entry.

    Canonical registry rows ``custom_llm``, ``openai``, ``anthropic``, and ``gemini`` are copied
    into typed slices for brains and for documentation; the full registry remains in ``providers``.

    Attributes:
        episode_max_cycles: Hard cap on replan cycles per episode.
        planner_allowed_tools: Tuple of tool names embedded in the system prompt.
        allowed_tools_policy: Set view used for fast membership checks in the engine.
        system_prompt: Fully rendered planner system string (placeholder expanded).
        fake_brain_first_tool: First tool name for :class:`~chief.brain.FakeBrain`.
        fake_brain_second_tool: Second tool name for :class:`~chief.brain.FakeBrain`.
        llm_integration_enabled: Gate for optional live-LLM tests.
        custom_llm: Slice for :class:`~chief.llm.providers.custom_chat_completions.CustomChatCompletionsBrain`.
        openai: Slice for :class:`~chief.llm.providers.openai.OpenAiChatCompletionsBrain`.
        anthropic: Slice for :class:`~chief.llm.providers.anthropic_messages.AnthropicMessagesBrain`.
        gemini: Slice for :class:`~chief.llm.providers.google_generative.GeminiGenerateContentBrain`.
        providers: All registry rows in load order (tuple for immutability).
        providers_by_id: Map from provider ``id`` to :class:`~chief.config.providers_registry.ProviderRecord`.
        default_provider_id: Normalized ``[chief].default_provider`` (``fake`` or a registry id).
        models: All catalog rows (tuple for immutability).
        models_by_id: Map from model catalog ``id`` to :class:`~chief.config.models_registry.ModelRecord`.
        default_model_id: Optional ``[chief].default_model`` catalog id (normalized); ``None`` if unset.
        serve_socket_path: Resolved Unix socket path for ``chief serve``.
    """

    episode_max_cycles: int
    planner_allowed_tools: tuple[str, ...]
    allowed_tools_policy: frozenset[str]
    system_prompt: str
    fake_brain_first_tool: str
    fake_brain_second_tool: str
    llm_integration_enabled: bool
    custom_llm: CustomLlmSlice
    openai: OpenAiSlice
    anthropic: AnthropicSlice
    gemini: GeminiSlice
    providers: tuple[ProviderRecord, ...]
    providers_by_id: dict[str, ProviderRecord]
    default_provider_id: str
    models: tuple[ModelRecord, ...]
    models_by_id: dict[str, ModelRecord]
    default_model_id: str | None
    serve_socket_path: Path


def openai_wire_model_and_json_mode(
    rt: RuntimeConfig,
    provider_record: ProviderRecord,
) -> tuple[str, bool]:
    """Resolve API model id and Chat Completions ``json_mode`` for a provider row.

    If ``rt.default_model_id`` is set and its catalog entry references this provider
    (``ModelRecord.provider_id == provider_record.id``), returns that entry's ``model`` and
    ``json_mode``. Otherwise returns the provider registry ``model`` and ``json_mode=False``.

    Args:
        rt: Frozen runtime snapshot (includes model catalog).
        provider_record: Row from the provider registry (must be ``custom_llm`` or ``openai``).

    Returns:
        ``(api_model_string, request_json_mode)`` for OpenAI-shaped Chat Completions.

    Raises:
        ValueError: If ``provider_record.kind`` is not ``custom_llm`` or ``openai``.
    """
    if provider_record.kind not in ("custom_llm", "openai"):
        raise ValueError(
            f"openai_wire_model_and_json_mode applies to custom_llm/openai, got {provider_record.kind!r}"
        )
    mid = rt.default_model_id
    if mid:
        spec = rt.models_by_id.get(mid)
        if spec is not None and spec.provider_id == provider_record.id:
            return spec.model, spec.json_mode
    return provider_record.model, False


def _expect_provider(by_id: dict[str, ProviderRecord], pid: str, kind: str) -> ProviderRecord:
    """Return the registry row for ``pid`` after checking its ``kind``.

    Args:
        by_id: Map built from :func:`chief.config.providers_registry.load_merged_providers`.
        pid: Canonical provider id (e.g. ``custom_llm``).
        kind: Expected ``ProviderRecord.kind`` for that id.

    Returns:
        The matching :class:`~chief.config.providers_registry.ProviderRecord`.

    Raises:
        ValueError: If the id is missing or ``kind`` does not match.
    """
    r = by_id.get(pid)
    if r is None:
        raise ValueError(
            f"provider registry missing required id {pid!r} (needed for runtime.{pid} slice)"
        )
    if r.kind != kind:
        raise ValueError(f"provider id {pid!r} must have kind {kind!r}, got {r.kind!r}")
    return r


def build_runtime_config() -> RuntimeConfig:
    """Load merged TOML and provider JSON once and return a :class:`RuntimeConfig`.

    Validates that the planner template contains ``{allowed_tools}``, that
    ``[chief].default_provider`` names ``fake`` or an existing registry id, that
    ``[chief].default_model`` (if set) exists in the model catalog and references a known
    ``provider_id``, that every catalog row references a known provider id, and that required
    canonical provider rows exist with correct ``kind`` and (for Anthropic) ``max_tokens`` /
    ``api_version``.

    Returns:
        Frozen snapshot safe to pass across CLI, IPC, and brains.

    Raises:
        ValueError: On missing placeholders, unknown default provider, invalid model catalog,
            unknown ``default_model``, missing canonical provider rows, wrong ``kind``, or
            incomplete Anthropic fields.
    """
    cfg = load_merged_config()
    allowed = planner_allowed_tools(cfg)
    tpl = str_from(cfg, "planner", "system_prompt_template")
    tools_csv = ", ".join(sorted(allowed))
    if "{allowed_tools}" not in tpl:
        raise ValueError("[planner].system_prompt_template must contain literal {allowed_tools} placeholder")
    system_prompt = tpl.replace("{allowed_tools}", tools_csv)

    providers = load_merged_providers()
    by_id = {p.id: p for p in providers}

    default_pid = str_from(cfg, "chief", "default_provider").strip().lower()
    if default_pid != "fake" and default_pid not in by_id:
        raise ValueError(
            f"[chief].default_provider={default_pid!r} not found in provider registry "
            f"(known ids: {', '.join(sorted(by_id))})"
        )
    if default_pid != "fake":
        dp = by_id[default_pid]
        if not dp.enabled:
            raise ValueError(
                f"[chief].default_provider={default_pid!r} is disabled in the provider registry "
                f"(enabled=false)"
            )

    models = load_merged_models()
    models_by_id = {m.id: m for m in models}
    default_model_raw = optional_str_from(cfg, "chief", "default_model")
    default_model_id = default_model_raw.strip().lower() if default_model_raw else None
    if default_model_id is not None:
        if default_model_id not in models_by_id:
            raise ValueError(
                f"[chief].default_model={default_model_id!r} not found in model catalog "
                f"(known ids: {', '.join(sorted(models_by_id)) or '(empty)'})"
            )
        dm = models_by_id[default_model_id]
        if dm.provider_id not in by_id:
            raise ValueError(
                f"model catalog {default_model_id!r} references unknown provider_id {dm.provider_id!r}"
            )

    for m in models:
        if m.provider_id not in by_id:
            raise ValueError(
                f"model catalog id {m.id!r} references unknown provider_id {m.provider_id!r}"
            )

    c_rec = _expect_provider(by_id, _CANONICAL_CUSTOM_LLM, "custom_llm")
    custom = CustomLlmSlice(
        base_url=c_rec.base_url,
        model=c_rec.model,
        api_key=c_rec.api_key,
        timeout_seconds=c_rec.timeout_seconds,
    )

    o_rec = _expect_provider(by_id, _CANONICAL_OPENAI, "openai")
    oa = OpenAiSlice(
        vendor_api_base=o_rec.base_url,
        model=o_rec.model,
        api_key=o_rec.api_key or "",
        timeout_seconds=o_rec.timeout_seconds,
    )

    a_rec = _expect_provider(by_id, _CANONICAL_ANTHROPIC, "anthropic")
    if a_rec.max_tokens is None or a_rec.api_version is None:
        raise ValueError("anthropic provider record requires max_tokens and api_version")
    ant = AnthropicSlice(
        vendor_api_base=a_rec.base_url,
        api_version=a_rec.api_version,
        api_key=a_rec.api_key or "",
        model=a_rec.model,
        timeout_seconds=a_rec.timeout_seconds,
        max_tokens=a_rec.max_tokens,
    )

    g_rec = _expect_provider(by_id, _CANONICAL_GEMINI, "gemini")
    gem = GeminiSlice(
        vendor_api_base=g_rec.base_url,
        api_key=g_rec.api_key or "",
        model=g_rec.model,
        timeout_seconds=g_rec.timeout_seconds,
    )

    sock_raw = optional_str_from(cfg, "serve", "socket_path")
    serve_sock = (
        Path(sock_raw).expanduser().resolve()
        if sock_raw
        else default_serve_socket_path()
    )

    return RuntimeConfig(
        episode_max_cycles=episode_max_cycles(cfg),
        planner_allowed_tools=allowed,
        allowed_tools_policy=frozenset(allowed),
        system_prompt=system_prompt,
        fake_brain_first_tool=str_from(cfg, "fake_brain", "first_tool"),
        fake_brain_second_tool=str_from(cfg, "fake_brain", "second_tool"),
        llm_integration_enabled=_llm_integration_enabled_from_mapping(cfg),
        custom_llm=custom,
        openai=oa,
        anthropic=ant,
        gemini=gem,
        providers=providers,
        providers_by_id=by_id,
        default_provider_id=default_pid,
        models=models,
        models_by_id=models_by_id,
        default_model_id=default_model_id,
        serve_socket_path=serve_sock,
    )
