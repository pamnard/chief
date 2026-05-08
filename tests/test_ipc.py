"""Tests for NDJSON IPC, protocol parsing, and serve dispatch (Unix socket)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from pathlib import Path

import pytest

from chief.config import RuntimeConfig, build_runtime_config
from chief.domain import EpisodeStatus
from chief.ipc.protocol import parse_request_line, response_error
from chief.ipc.server import _ChiefIpcSession, _dispatch_line, run_serve_forever

unix_only = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="Unix domain sockets required",
)


def test_parse_orchestrator_minimal() -> None:
    """Orchestrator target should accept omitted provider (null in JSON)."""
    raw = json.dumps(
        {"v": 1, "session_id": "s1", "target": "orchestrator", "text": "ping", "provider": None}
    )
    req = parse_request_line(raw)
    assert req.target == "orchestrator"
    assert req.provider is None
    assert req.text == "ping"
    assert req.session_id == "s1"


def test_parse_subagent_requires_provider() -> None:
    """Subagent routing must name a concrete provider."""
    raw = json.dumps({"v": 1, "session_id": "x", "target": "subagent", "text": "t", "provider": None})
    with pytest.raises(ValueError, match="subagent requires provider"):
        parse_request_line(raw)


def test_parse_orchestrator_rejects_provider_field() -> None:
    """Orchestrator must not carry an explicit provider (server default applies)."""
    raw = json.dumps({"v": 1, "session_id": "x", "target": "orchestrator", "text": "t", "provider": "fake"})
    with pytest.raises(ValueError, match="orchestrator must omit provider"):
        parse_request_line(raw)


def test_parse_bad_version() -> None:
    """Unknown protocol version is rejected."""
    raw = json.dumps({"v": 9, "session_id": "x", "target": "orchestrator", "text": "t"})
    with pytest.raises(ValueError, match="unsupported v"):
        parse_request_line(raw)


@pytest.fixture
def runtime() -> RuntimeConfig:
    """Single merged config snapshot."""
    return build_runtime_config()


async def test_dispatch_orchestrator_fake_completes(
    isolated_episode_logs: Path, runtime: RuntimeConfig
) -> None:
    """Dispatch should run FakeBrain path for orchestrator default."""
    sess = _ChiefIpcSession()
    line = json.dumps(
        {"v": 1, "session_id": "sid", "target": "orchestrator", "text": "hello", "provider": None}
    )
    out = await _dispatch_line(runtime, default_provider="fake", sessions=sess, line=line)
    assert out["ok"] is True
    ep = out["episode"]
    assert ep["status"] == EpisodeStatus.COMPLETED.value
    assert "hello" in (ep.get("artifact") or "")


async def test_dispatch_subagent_fake(
    isolated_episode_logs: Path, runtime: RuntimeConfig
) -> None:
    """Explicit subagent provider should bypass server default."""
    sess = _ChiefIpcSession()
    line = json.dumps(
        {"v": 1, "session_id": "sid", "target": "subagent", "text": "hello", "provider": "fake"}
    )
    out = await _dispatch_line(runtime, default_provider="custom_llm", sessions=sess, line=line)
    assert out["ok"] is True
    assert out["episode"]["status"] == EpisodeStatus.COMPLETED.value


async def test_dispatch_bad_json_returns_error(runtime: RuntimeConfig) -> None:
    """Malformed line should yield ok=false without raising."""
    sess = _ChiefIpcSession()
    out = await _dispatch_line(runtime, default_provider="fake", sessions=sess, line="not-json")
    assert out["ok"] is False
    assert out["error"]


def test_response_error_shape() -> None:
    """Error helper should match v1 envelope."""
    r = response_error("oops")
    assert r["ok"] is False
    assert r["error"] == "oops"
    assert r["episode"] is None


@unix_only
@pytest.mark.asyncio
async def test_serve_one_roundtrip(
    isolated_episode_logs: Path, runtime: RuntimeConfig, tmp_path: Path
) -> None:
    """Live Unix server: one NDJSON line in, one response out."""
    sock = tmp_path / "chief-test.sock"
    task = asyncio.create_task(
        run_serve_forever(runtime, default_provider="fake", socket_path=sock),
    )
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    try:
        for _ in range(200):
            if sock.exists():
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_unix_connection(str(sock)),
                        timeout=0.05,
                    )
                    break
                except (FileNotFoundError, OSError, asyncio.TimeoutError):
                    await asyncio.sleep(0.01)
                    continue
            await asyncio.sleep(0.01)
        else:
            pytest.fail("could not connect to chief serve test socket")

        assert reader is not None
        assert writer is not None
        payload = json.dumps(
            {
                "v": 1,
                "session_id": "pytest",
                "target": "orchestrator",
                "text": "hi",
                "provider": None,
            }
        )
        writer.write((payload + "\n").encode("utf-8"))
        await writer.drain()
        line_b = await asyncio.wait_for(reader.readline(), timeout=5.0)
        obj = json.loads(line_b.decode("utf-8"))
        assert obj["ok"] is True
        assert obj["episode"]["status"] == EpisodeStatus.COMPLETED.value
    finally:
        if writer is not None:
            writer.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                await writer.wait_closed()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
