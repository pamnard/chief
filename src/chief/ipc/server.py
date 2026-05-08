"""Async Unix-socket server that runs :func:`~chief.engine.run_episode` per request line."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import Any

from chief.brain_select import select_brain
from chief.config import RuntimeConfig
from chief.engine import run_episode
from chief.ipc.protocol import parse_request_line, response_error, response_ok
from chief.llm.readiness import ensure_llm_ready_or_raise
from chief.tools import build_registry


class _ChiefIpcSession:
    """Per-server state: session transcript keys (reserved for future turns)."""

    def __init__(self) -> None:
        """Initialize empty in-memory session transcripts."""
        self._sessions: dict[str, list[str]] = {}

    def record_user_line(self, session_id: str, text: str) -> None:
        """Append user text to the in-memory transcript for ``session_id``.

        Args:
            session_id: Client-supplied session key.
            text: User message for this request.
        """
        self._sessions.setdefault(session_id, []).append(text)


async def _dispatch_line(
    runtime: RuntimeConfig,
    *,
    default_provider: str,
    sessions: _ChiefIpcSession,
    line: str,
) -> dict[str, Any]:
    """Parse one NDJSON line, run an episode, return a v1 response dict.

    Args:
        runtime: Frozen configuration for tools and limits.
        default_provider: Planner id when ``target`` is ``orchestrator``.
        sessions: Server-side session book-keeping.
        line: One UTF-8 JSON line from the client.

    Returns:
        Response dict suitable for :func:`json.dumps`.
    """
    try:
        req = parse_request_line(line)
    except ValueError as exc:
        return response_error(str(exc))
    try:
        if req.target == "subagent":
            brain = select_brain(req.provider or "", runtime)
        else:
            brain = select_brain(default_provider, runtime)
    except ValueError as exc:
        return response_error(str(exc))
    sessions.record_user_line(req.session_id, req.text)
    tools = build_registry(runtime)
    episode = await run_episode(
        req.text,
        runtime=runtime,
        brain=brain,
        tools=tools,
        max_cycles=None,
    )
    return response_ok(req.session_id, episode)


async def run_serve_forever(
    runtime: RuntimeConfig,
    *,
    default_provider: str,
    socket_path: Path | None = None,
) -> None:
    """Bind a Unix socket and serve NDJSON request/response lines until interrupted.

    Removes an existing socket path before bind. Writes ``chief-serve.pid`` beside the
    socket and deletes it on exit.

    Args:
        runtime: Merged configuration (socket default from ``serve_socket_path``).
        default_provider: Planner id when the client sets ``target`` to ``orchestrator``.
        socket_path: Override socket path; ``None`` uses ``runtime.serve_socket_path``.

    Raises:
        OSError: If the socket cannot be created or bound.
        chief.llm.readiness.LlmNotReadyError: If ``default_provider`` is not ``fake`` and readiness
            probe fails.
    """
    if default_provider.strip().lower() != "fake":
        await ensure_llm_ready_or_raise(runtime, default_provider)

    sock = socket_path if socket_path is not None else runtime.serve_socket_path
    sock.parent.mkdir(parents=True, exist_ok=True)
    if sock.exists():
        sock.unlink()
    state = _ChiefIpcSession()

    async def _client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve sequential NDJSON request lines on one accepted connection."""
        try:
            while True:
                line_b = await reader.readline()
                if not line_b:
                    break
                line = line_b.decode("utf-8").strip()
                if not line:
                    continue
                out = await _dispatch_line(
                    runtime,
                    default_provider=default_provider,
                    sessions=state,
                    line=line,
                )
                writer.write((json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    server = await asyncio.start_unix_server(_client, path=str(sock))
    pid_path = sock.parent / "chief-serve.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        """Signal handler: stop accepting and unwind ``serve_forever``."""
        server.close()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)
    try:
        async with server:
            print(f"chief serve listening on {sock}", flush=True)
            await server.serve_forever()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        pid_path.unlink(missing_ok=True)
        if sock.exists():
            sock.unlink()
