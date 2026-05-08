"""Local IPC for ``chief serve`` and ``chief chat`` (Unix socket, NDJSON v1)."""

from chief.ipc.client import run_chat_client
from chief.ipc.protocol import ChatRequest, parse_request_line, response_error, response_ok
from chief.ipc.server import run_serve_forever

__all__ = [
    "ChatRequest",
    "parse_request_line",
    "response_error",
    "response_ok",
    "run_chat_client",
    "run_serve_forever",
]
