"""Unit tests for planner JSON parsing (no network)."""

from __future__ import annotations

import pytest

from chief.domain import FinalIntent, ToolIntent
from chief.llm import IntentPayloadError, parse_intent_from_model_json


def test_parse_tool_intent() -> None:
    raw = '{"intent_type":"tool","tool":"echo","args":{"text":"hi"}}'
    intent = parse_intent_from_model_json(raw)
    assert isinstance(intent, ToolIntent)
    assert intent.tool == "echo"
    assert intent.args == {"text": "hi"}


def test_parse_final_intent() -> None:
    raw = '{"intent_type":"final","message":"done"}'
    intent = parse_intent_from_model_json(raw)
    assert isinstance(intent, FinalIntent)
    assert intent.message == "done"


def test_parse_strips_json_fence() -> None:
    raw = '```json\n{"intent_type":"final","message":"x"}\n```'
    intent = parse_intent_from_model_json(raw)
    assert isinstance(intent, FinalIntent)
    assert intent.message == "x"


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "{}",
        '{"intent_type":"tool"}',
        '{"intent_type":"tool","tool":"","args":{}}',
        '{"intent_type":"final","message":""}',
        '{"intent_type":"noop"}',
    ],
)
def test_parse_errors(bad: str) -> None:
    with pytest.raises(IntentPayloadError):
        parse_intent_from_model_json(bad)
