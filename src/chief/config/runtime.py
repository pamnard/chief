"""Process-wide configuration snapshot: load once, reuse until process exit."""

from __future__ import annotations

from dataclasses import dataclass

from chief.config.loader import (
    bool_from,
    episode_max_cycles,
    float_from,
    int_from,
    llm_integration_enabled as _llm_integration_enabled_from_mapping,
    load_merged_config,
    optional_str_from,
    planner_allowed_tools,
    str_from,
)


@dataclass(frozen=True, slots=True)
class CustomLlmSlice:
    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float
    json_mode: bool


@dataclass(frozen=True, slots=True)
class OpenAiSlice:
    vendor_api_base: str
    model: str
    api_key: str
    timeout_seconds: float
    json_mode: bool


@dataclass(frozen=True, slots=True)
class AnthropicSlice:
    vendor_api_base: str
    api_version: str
    api_key: str
    model: str
    timeout_seconds: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class GeminiSlice:
    vendor_api_base: str
    api_key: str
    model: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Immutable snapshot built at process entry (CLI, tests, future server lifespan)."""

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


def build_runtime_config() -> RuntimeConfig:
    """Merge TOML + env once and materialize typed fields (no further disk reads)."""
    cfg = load_merged_config()
    allowed = planner_allowed_tools(cfg)
    tpl = str_from(cfg, "planner", "system_prompt_template")
    tools_csv = ", ".join(sorted(allowed))
    if "{allowed_tools}" not in tpl:
        raise ValueError("[planner].system_prompt_template must contain literal {allowed_tools} placeholder")
    system_prompt = tpl.replace("{allowed_tools}", tools_csv)

    custom = CustomLlmSlice(
        base_url=str_from(cfg, "custom_llm", "base_url"),
        model=str_from(cfg, "custom_llm", "model"),
        api_key=optional_str_from(cfg, "custom_llm", "api_key"),
        timeout_seconds=float_from(cfg, "custom_llm", "timeout_seconds"),
        json_mode=bool_from(cfg, "custom_llm", "json_mode"),
    )
    oa = OpenAiSlice(
        vendor_api_base=str_from(cfg, "openai", "vendor_api_base"),
        model=str_from(cfg, "openai", "model"),
        api_key=str_from(cfg, "openai", "api_key"),
        timeout_seconds=float_from(cfg, "openai", "timeout_seconds"),
        json_mode=bool_from(cfg, "openai", "json_mode"),
    )
    ant = AnthropicSlice(
        vendor_api_base=str_from(cfg, "anthropic", "vendor_api_base"),
        api_version=str_from(cfg, "anthropic", "api_version"),
        api_key=str_from(cfg, "anthropic", "api_key"),
        model=str_from(cfg, "anthropic", "model"),
        timeout_seconds=float_from(cfg, "anthropic", "timeout_seconds"),
        max_tokens=int_from(cfg, "anthropic", "max_tokens"),
    )
    gem = GeminiSlice(
        vendor_api_base=str_from(cfg, "gemini", "vendor_api_base"),
        api_key=str_from(cfg, "gemini", "api_key"),
        model=str_from(cfg, "gemini", "model"),
        timeout_seconds=float_from(cfg, "gemini", "timeout_seconds"),
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
    )
