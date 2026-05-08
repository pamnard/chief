"""Public package surface for ``chief``: episode runtime types and ``run_episode``.

This package implements the v0 agent episode loop. Imports here
are intentionally minimal so downstream code can depend on a stable API.

``run_episode`` is **async**; build :class:`~chief.config.runtime.RuntimeConfig` once (e.g.
:func:`~chief.config.runtime.build_runtime_config`), then ``asyncio.run(run_episode(..., runtime=rt, ...))``
or ``await`` inside an async task.
"""

__version__ = "0.4.0"

from chief.domain import Episode, EpisodeStatus, Observation
from chief.engine import run_episode
from chief.llm import (
    AnthropicMessagesBrain,
    CustomChatCompletionsBrain,
    GeminiGenerateContentBrain,
    OpenAiChatCompletionsBrain,
)
from chief.config import RuntimeConfig, build_runtime_config

__all__ = [
    "__version__",
    "AnthropicMessagesBrain",
    "CustomChatCompletionsBrain",
    "Episode",
    "EpisodeStatus",
    "GeminiGenerateContentBrain",
    "Observation",
    "OpenAiChatCompletionsBrain",
    "RuntimeConfig",
    "build_runtime_config",
    "run_episode",
]
