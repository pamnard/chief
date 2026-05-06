"""Public package surface for ``chief``: episode runtime types and ``run_episode``.

This package implements the v0 agent episode loop (see internal SPEC). Imports here
are intentionally minimal so downstream code can depend on a stable API.
"""

__version__ = "0.2.0"

from chief.domain import Episode, EpisodeStatus, Observation
from chief.engine import run_episode
from chief.llm import HttpChatCompletionsBrain, OpenAIChatBrain

__all__ = [
    "__version__",
    "Episode",
    "EpisodeStatus",
    "HttpChatCompletionsBrain",
    "Observation",
    "OpenAIChatBrain",
    "run_episode",
]
