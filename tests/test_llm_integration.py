"""Optional integration tests against a custom ``/v1/chat/completions`` gateway.

Configure the ``custom_llm`` entry in the provider registry
(``defaults.providers.json`` + optional ``~/.config/chief/providers.json``).
Enable with ``CHIEF_TEST_LLM=1`` or ``[test].enable_llm_integration = true``.
"""

from __future__ import annotations

import pytest

from chief.domain import FinalIntent, ToolIntent
from chief.llm import CustomChatCompletionsBrain
from chief.memory import MemorySession
from chief.config import build_runtime_config


@pytest.mark.integration
async def test_custom_chat_completion_returns_intent() -> None:
    """One live ``/v1/chat/completions`` call; skipped unless integration flag is on."""
    runtime = build_runtime_config()
    if not runtime.llm_integration_enabled:
        pytest.skip("set CHIEF_TEST_LLM=1 or [test].enable_llm_integration=true in merged config")

    if runtime.providers_by_id.get("custom_llm") is None:
        pytest.skip("registry has no 'custom_llm' provider")

    brain = CustomChatCompletionsBrain.from_runtime(runtime)
    mem = MemorySession()
    intent = await brain.reason(
        mem,
        'Return exactly this JSON object and nothing else: '
        '{"intent_type":"final","message":"ok-from-llm"}',
    )
    assert isinstance(intent, (FinalIntent, ToolIntent))
