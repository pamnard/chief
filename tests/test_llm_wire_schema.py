"""Unit tests for wire schema codecs (no network)."""

from __future__ import annotations

from chief.llm.errors import ChatCompletionTransportError
from chief.llm.schema import anthropic_messages as am
from chief.llm.schema import google_generative as gg
from chief.llm.types import ModelRef


def test_anthropic_messages_url() -> None:
    assert am.messages_url("https://api.anthropic.com/v1") == "https://api.anthropic.com/v1/messages"


def test_anthropic_extract_assistant_text() -> None:
    text = am.extract_assistant_text(
        {"content": [{"type": "text", "text": '{"intent_type":"final","message":"x"}'}]},
    )
    assert "intent_type" in text


def test_anthropic_extract_bad_shape() -> None:
    try:
        am.extract_assistant_text({})
    except ChatCompletionTransportError:
        return
    raise AssertionError("expected ChatCompletionTransportError")


def test_gemini_generate_content_url() -> None:
    u = gg.generate_content_url(
        "https://generativelanguage.googleapis.com/v1beta",
        ModelRef(id="gemini-1.5-flash"),
    )
    assert u.endswith("models/gemini-1.5-flash:generateContent")


def test_gemini_generate_content_url_strips_models_prefix() -> None:
    u = gg.generate_content_url(
        "https://generativelanguage.googleapis.com/v1beta",
        ModelRef(id="models/gemini-pro"),
    )
    assert "models/gemini-pro:generateContent" in u


def test_gemini_extract_assistant_text() -> None:
    text = gg.extract_assistant_text(
        {"candidates": [{"content": {"parts": [{"text": '{"intent_type":"final","message":"y"}'}]}}]},
    )
    assert "intent_type" in text
