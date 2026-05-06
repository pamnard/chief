"""Optional integration tests against any OpenAI-compatible Chat Completions API.

For local Ollama: ``ollama serve`` with OpenAI compatibility (default base in code).
Enable with ``CHIEF_TEST_LLM=1``.
"""

from __future__ import annotations

import os

import pytest

from chief.domain import FinalIntent, ToolIntent
from chief.memory import MemorySession
from chief.llm import HttpChatCompletionsBrain


@pytest.mark.integration
def test_openai_chat_completion_returns_intent() -> None:
    """One live ``/v1/chat/completions`` call; skipped unless ``CHIEF_TEST_LLM=1``."""
    if os.environ.get("CHIEF_TEST_LLM") != "1":
        pytest.skip("set CHIEF_TEST_LLM=1 to exercise LLM provider")

    brain = HttpChatCompletionsBrain.from_env()
    mem = MemorySession()
    intent = brain.reason(
        mem,
        'Return exactly this JSON object and nothing else: '
        '{"intent_type":"final","message":"ok-from-llm"}',
    )
    assert isinstance(intent, (FinalIntent, ToolIntent))
