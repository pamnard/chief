"""Async NDJSON client for interactive ``chief chat``."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

Target = Literal["orchestrator", "subagent"]


async def run_chat_client(
    socket_path: Path,
    *,
    session_id: str,
    target: Target,
    provider: str | None,
    verbose: bool = False,
) -> None:
    """Connect to ``chief serve`` and relay stdin lines to JSON requests until EOF.

    Each non-empty stdin line becomes one ``text`` field. For ``target=subagent``,
    ``provider`` must be set (``fake`` or a registry id). For ``orchestrator``,
    ``provider`` must be ``None`` (server default).

    On success, stdout prints only the episode ``artifact`` (assistant text) by
    default. With ``verbose=True``, a one-line summary (status, id, ticks) is
    printed before the artifact.

    Args:
        socket_path: Path to the running server's Unix socket.
        session_id: Logical session id sent on every line.
        target: Routing target for all lines in this REPL.
        provider: Subagent provider id when ``target`` is ``subagent``.
        verbose: If ``True``, print episode summary before artifact text.

    Raises:
        OSError: If the socket is missing or not connectable.
        RuntimeError: If ``target`` / ``provider`` combination is invalid for chat.
    """
    if target == "subagent" and not provider:
        raise RuntimeError("chat client: subagent target requires --provider")
    if target == "orchestrator" and provider is not None:
        raise RuntimeError("chat client: orchestrator target must not set --provider")

    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if line == "":
                break
            text = line.rstrip("\n").strip()
            if not text:
                continue
            req: dict[str, Any] = {
                "v": 1,
                "session_id": session_id,
                "target": target,
                "text": text,
                "provider": provider,
            }
            writer.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
            await writer.drain()
            resp_b = await reader.readline()
            if not resp_b:
                print("chief chat: server closed connection", file=sys.stderr)
                break
            try:
                obj = json.loads(resp_b.decode("utf-8"))
            except json.JSONDecodeError as exc:
                print(f"chief chat: bad response json: {exc}", file=sys.stderr)
                continue
            if obj.get("ok"):
                ep = obj.get("episode") or {}
                if verbose:
                    print(
                        f"status={ep.get('status')} id={ep.get('id')} ticks={ep.get('ticks')}",
                        flush=True,
                    )
                if ep.get("artifact"):
                    print(ep["artifact"], flush=True)
            else:
                print(f"error: {obj.get('error')}", file=sys.stderr)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
