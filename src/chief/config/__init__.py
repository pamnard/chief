"""Configuration package for ``chief``.

Exposes TOML merge helpers (:mod:`chief.config.loader`), typed runtime snapshots
(:mod:`chief.config.runtime`), provider registry types (:mod:`chief.config.providers_registry`),
and re-exports the public API for convenient ``from chief.config import …`` imports.
"""

from chief.config.loader import (
    bool_from,
    episode_max_cycles,
    float_from,
    int_from,
    load_merged_config,
    llm_integration_enabled,
    optional_str_from,
    planner_allowed_tools,
    section,
    str_from,
)
from chief.config.models_registry import ModelRecord, load_merged_models, technical_model_candidates
from chief.config.providers_registry import ProviderRecord, load_merged_providers
from chief.config.runtime import (
    AnthropicSlice,
    CustomLlmSlice,
    GeminiSlice,
    OpenAiSlice,
    RuntimeConfig,
    build_runtime_config,
    openai_wire_model_and_json_mode,
)

__all__ = [
    "AnthropicSlice",
    "CustomLlmSlice",
    "GeminiSlice",
    "ModelRecord",
    "OpenAiSlice",
    "ProviderRecord",
    "RuntimeConfig",
    "bool_from",
    "build_runtime_config",
    "episode_max_cycles",
    "float_from",
    "int_from",
    "load_merged_config",
    "load_merged_models",
    "load_merged_providers",
    "llm_integration_enabled",
    "openai_wire_model_and_json_mode",
    "optional_str_from",
    "planner_allowed_tools",
    "section",
    "str_from",
    "technical_model_candidates",
]
