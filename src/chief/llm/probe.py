"""Lightweight HTTP probes for provider registry rows (Phase 2 readiness).

Each probe uses a short timeout and a minimal vendor-specific request: ``GET /v1/models`` when
available for OpenAI-shaped bases, otherwise a tiny Chat Completions / Messages / generateContent
call with ``max_tokens`` / ``maxOutputTokens`` capped at 1.
"""

from __future__ import annotations

import json
import os
from typing import Any, Final

import httpx

from chief.config.providers_registry import ProviderRecord
from chief.config.runtime import RuntimeConfig, openai_wire_model_and_json_mode
from chief.llm.schema import anthropic_messages as am
from chief.llm.schema import google_generative as gg
from chief.llm.schema import openai_chat_completions as occ
from chief.llm.types import ModelRef

_ENV_PROBE_TIMEOUT: Final = "CHIEF_LLM_PROBE_TIMEOUT"
_DEFAULT_PROBE_TIMEOUT: Final = 5.0


def probe_http_timeout_seconds() -> float:
    """Return the hard cap (seconds) for a single provider probe HTTP exchange.

    Returns:
        Positive float from ``CHIEF_LLM_PROBE_TIMEOUT`` or a small default.
    """
    raw = os.environ.get(_ENV_PROBE_TIMEOUT, "").strip()
    if not raw:
        return _DEFAULT_PROBE_TIMEOUT
    try:
        v = float(raw)
    except ValueError:
        return _DEFAULT_PROBE_TIMEOUT
    return v if v > 0.1 else 0.1


def _auth_headers_bearer(api_key: str | None) -> dict[str, str]:
    """Build optional ``Authorization: Bearer`` headers for OpenAI-shaped HTTP calls.

    Args:
        api_key: Raw bearer token; whitespace-only treated as absent.

    Returns:
        Either ``{"Authorization": "Bearer …"}`` or an empty dict.
    """
    if api_key and api_key.strip():
        return {"Authorization": f"Bearer {api_key.strip()}"}
    return {}


async def _probe_openai_chat_wire(
    record: ProviderRecord,
    *,
    api_model: str,
    json_mode: bool,
) -> bool:
    """Probe an OpenAI Chat Completions-compatible base with a cheap request.

    Tries ``GET {base}/models`` first; on non-definitive outcome falls back to a one-token
    ``POST …/chat/completions`` using :mod:`chief.llm.schema.openai_chat_completions`.

    Args:
        record: Provider row (``base_url``, ``api_key``, etc.).
        api_model: Model id to embed in the fallback completion body.
        json_mode: Whether to set JSON object mode on the fallback body (matches runtime behavior).

    Returns:
        ``True`` on HTTP 200 from ``/models`` or a successful completion POST.
    """
    base = record.base_url.rstrip("/")
    timeout = httpx.Timeout(probe_http_timeout_seconds())
    headers_json = {"Content-Type": "application/json"}
    headers_json.update(_auth_headers_bearer(record.api_key))
    headers_get = _auth_headers_bearer(record.api_key)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            rm = await client.get(f"{base}/models", headers=headers_get)
            if rm.status_code == 200:
                return True
            if rm.status_code in (401, 403):
                return False
        except httpx.RequestError:
            pass

        model = ModelRef(id=api_model)
        messages = occ.build_messages("ping", "ping")
        payload: dict[str, Any] = occ.build_request_body(model, messages, json_mode=json_mode)
        payload["max_tokens"] = 1
        url = occ.chat_completions_url(base)
        try:
            resp = await client.post(
                url,
                content=json.dumps(payload).encode("utf-8"),
                headers=headers_json,
            )
        except httpx.RequestError:
            return False
        return resp.is_success


async def _probe_anthropic(record: ProviderRecord) -> bool:
    """Probe Anthropic Messages with a single user turn and ``max_tokens`` capped at 1.

    Args:
        record: Provider row with ``base_url``, ``model``, ``api_key``, ``api_version``,
            ``max_tokens``.

    Returns:
        ``True`` if the HTTP response is successful (transport errors yield ``False``).

    Note:
        Requires ``record.api_version`` and ``record.max_tokens`` (validated at registry load).
    """
    assert record.api_version is not None and record.max_tokens is not None
    base = record.base_url.rstrip("/")
    url = am.messages_url(base)
    model = ModelRef(id=record.model)
    messages = am.build_messages("ping")
    body = am.build_request_body(
        model,
        messages,
        system="",
        max_tokens=min(1, record.max_tokens),
    )
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": record.api_version,
        "x-api-key": record.api_key or "",
    }
    timeout = httpx.Timeout(probe_http_timeout_seconds())
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                content=json.dumps(body).encode("utf-8"),
                headers=headers,
            )
    except httpx.RequestError:
        return False
    return resp.is_success


async def _probe_gemini(record: ProviderRecord) -> bool:
    """Probe Google Generative Language ``generateContent`` with one output token.

    Builds URL via :func:`chief.llm.schema.google_generative.generate_content_url`, appends
    ``?key=`` from ``record.api_key``, and sends a minimal ``contents`` payload with
    ``generationConfig.maxOutputTokens`` set to 1.

    Args:
        record: ``gemini`` provider row (``base_url``, ``model``, ``api_key``).

    Returns:
        ``True`` if the HTTP response is successful; ``False`` on transport errors or non-success
        status.
    """
    model = ModelRef(id=record.model)
    path_url = gg.generate_content_url(record.base_url.rstrip("/"), model)
    url = f"{path_url}?key={record.api_key or ''}"
    body = gg.build_request_body(
        contents=gg.build_contents("ping"),
        system_instruction=None,
    )
    body["generationConfig"] = {"maxOutputTokens": 1}
    timeout = httpx.Timeout(probe_http_timeout_seconds())
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                content=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
    except httpx.RequestError:
        return False
    return resp.is_success


async def probe_provider_record(
    record: ProviderRecord,
    runtime: RuntimeConfig | None = None,
) -> bool:
    """Run one readiness probe for a single registry row.

    Args:
        record: Enabled provider with kind-specific fields already validated at registry load.
        runtime: When set, OpenAI-wire probes use :func:`~chief.config.runtime.openai_wire_model_and_json_mode`
            so the catalog ``default_model`` matches the probed API model.

    Returns:
        ``True`` if the endpoint responded successfully to the minimal probe.
    """
    if record.kind == "custom_llm" or record.kind == "openai":
        if runtime is not None:
            api_model, jm = openai_wire_model_and_json_mode(runtime, record)
        else:
            api_model, jm = record.model, False
        return await _probe_openai_chat_wire(record, api_model=api_model, json_mode=jm)
    if record.kind == "anthropic":
        return await _probe_anthropic(record)
    if record.kind == "gemini":
        return await _probe_gemini(record)
    return False
