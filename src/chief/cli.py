"""Command-line interface for running chief episodes.

The ``chief`` console script dispatches subcommands ``run``, ``serve``, ``chat``, and ``setup``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from chief.brain_select import select_brain
from chief.engine import run_episode
from chief.config import RuntimeConfig, build_runtime_config
from chief.config.config_check import (
    setup_hint_message,
    static_llm_config_issues,
    user_llm_overlay_present,
)
from chief.domain import Episode
from chief.ipc import run_chat_client, run_serve_forever
from chief.llm.readiness import LlmNotReadyError, ensure_llm_ready_or_raise
from chief.tools import build_registry


def _warn_missing_user_llm_overlay(cmd: str) -> None:
    """Print level-1 warning when the planner is not fake and no user overlay was found."""
    print(
        f"chief {cmd}: warning: LLM planner is active but no user configuration overlay was "
        f"detected; bundled defaults alone may be wrong for this machine. "
        f"{setup_hint_message()}",
        file=sys.stderr,
    )


def _exit_if_static_llm_invalid(runtime: RuntimeConfig, provider_id: str) -> int | None:
    """Return exit code 2 if static LLM config is invalid; otherwise ``None``."""
    issues = static_llm_config_issues(runtime, effective_provider_id=provider_id)
    if not issues:
        return None
    for line in issues:
        print(f"chief: configuration error: {line}", file=sys.stderr)
    print(f"chief: {setup_hint_message()}", file=sys.stderr)
    return 2


async def run_episode_with_readiness(
    runtime: RuntimeConfig,
    provider_id: str,
    task: str,
    max_cycles: int | None,
) -> Episode:
    """Run a single episode after :func:`~chief.llm.readiness.ensure_llm_ready_or_raise`.

    Args:
        runtime: Merged process configuration.
        provider_id: ``fake`` or a registry provider id.
        task: Episode task text.
        max_cycles: Replan cap; ``None`` uses ``runtime.episode_max_cycles``.

    Returns:
        Completed or stopped :class:`~chief.domain.Episode`.

    Raises:
        LlmNotReadyError: When the chosen provider is not ready for LLM planning.
        ValueError: Propagated from :func:`~chief.brain_select.select_brain` for bad ids.
    """
    await ensure_llm_ready_or_raise(runtime, provider_id)
    brain = select_brain(provider_id, runtime)
    cap = max_cycles if max_cycles is not None else runtime.episode_max_cycles
    return await run_episode(
        task,
        runtime=runtime,
        brain=brain,
        tools=build_registry(runtime),
        max_cycles=cap,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested subcommand.

    Args:
        argv: Argument list; ``None`` lets :mod:`argparse` read the process argv.

    Returns:
        Shell exit code: ``0`` on success, ``1`` on failure paths,
        ``2`` for CLI usage errors or ``setup`` without TTY.
    """
    parser = argparse.ArgumentParser(prog="chief")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a single episode (v0)")
    run_p.add_argument(
        "task",
        nargs="*",
        default=[],
        help="Task text (words after run)",
    )
    run_p.add_argument(
        "--provider",
        default=None,
        metavar="ID",
        help="Planner: fake or a registry id (default: [chief].default_provider)",
    )
    run_p.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Hard cap on replan cycles (default: runtime.episode_max_cycles)",
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary",
    )

    serve_p = sub.add_parser(
        "serve",
        help="Run long-lived Unix-socket server (NDJSON v1; default provider for orchestrator)",
    )
    serve_p.add_argument(
        "--provider",
        default=None,
        metavar="ID",
        help="Default planner when client uses target=orchestrator (default: [chief].default_provider)",
    )
    serve_p.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="Unix socket path (default: runtime.serve_socket_path / CHIEF_SERVE_SOCKET)",
    )

    chat_p = sub.add_parser("chat", help="Read stdin lines and send each to chief serve")
    chat_p.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="Unix socket path (default: runtime.serve_socket_path)",
    )
    chat_p.add_argument(
        "--session",
        default="cli",
        help="session_id sent with each line (default: cli)",
    )
    chat_p.add_argument(
        "--target",
        choices=("orchestrator", "subagent"),
        default="orchestrator",
        help="Protocol target field (subagent requires --provider)",
    )
    chat_p.add_argument(
        "--provider",
        default=None,
        metavar="ID",
        help="Planner for target=subagent: fake or registry id",
    )
    chat_p.add_argument(
        "--verbose",
        action="store_true",
        help="Print episode status line (id, ticks) before assistant text",
    )

    setup_p = sub.add_parser(
        "setup",
        help="Interactive bootstrap (TTY) for XDG config files",
    )
    setup_sub = setup_p.add_subparsers(dest="setup_cmd", required=True)
    setup_sub.add_parser(
        "providers",
        help="Write chief.toml + providers.json for canonical custom_llm",
    )

    args = parser.parse_args(argv)

    if args.cmd == "setup":
        from chief.setup_wizard import run_setup_providers

        if args.setup_cmd == "providers":
            return run_setup_providers()
        return 2

    if args.cmd == "run":
        task = " ".join(args.task).strip()
        runtime = build_runtime_config()
        pid = args.provider if args.provider is not None else runtime.default_provider_id
        if pid != "fake":
            if args.provider is None and not user_llm_overlay_present():
                _warn_missing_user_llm_overlay("run")
            bad = _exit_if_static_llm_invalid(runtime, pid)
            if bad is not None:
                return bad

        try:
            episode = asyncio.run(
                run_episode_with_readiness(
                    runtime,
                    pid,
                    task,
                    args.max_cycles,
                )
            )
        except LlmNotReadyError as exc:
            print(
                f"chief run: LLM not ready ({exc.state.value}): {exc.detail}",
                file=sys.stderr,
            )
            print(
                "chief run: fix configuration (interactive TTY: chief setup providers), "
                "or edit XDG chief.toml / providers.json.",
                file=sys.stderr,
            )
            return 1
        if args.json:
            print(
                json.dumps(
                    {
                        "id": episode.id,
                        "status": episode.status.value,
                        "artifact": episode.artifact,
                        "ticks": len(episode.ticks),
                        "task": task,
                        "provider": pid,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(
                f"episode={episode.id} status={episode.status.value} "
                f"ticks={len(episode.ticks)} provider={pid}"
            )
            if episode.artifact:
                print(f"artifact={episode.artifact}")
            for t in episode.ticks:
                print(f"  [{t.index}] cycle={t.cycle} phase={t.phase.value} {t.data}")

        return 0 if episode.status.value == "completed" else 1

    if args.cmd == "serve":
        runtime = build_runtime_config()
        default_p = args.provider if args.provider is not None else runtime.default_provider_id
        if default_p != "fake":
            if args.provider is None and not user_llm_overlay_present():
                _warn_missing_user_llm_overlay("serve")
            bad = _exit_if_static_llm_invalid(runtime, default_p)
            if bad is not None:
                return bad
        try:
            asyncio.run(
                run_serve_forever(
                    runtime,
                    default_provider=default_p,
                    socket_path=args.socket,
                )
            )
        except LlmNotReadyError as exc:
            print(
                f"chief serve: LLM not ready ({exc.state.value}): {exc.detail}",
                file=sys.stderr,
            )
            print(
                "chief serve: fix configuration (interactive TTY: chief setup providers), "
                "or edit XDG chief.toml / providers.json.",
                file=sys.stderr,
            )
            return 1
        except OSError as exc:
            print(f"chief serve: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "chat":
        if args.target == "subagent" and not args.provider:
            print("chief chat: --provider is required when --target subagent", file=sys.stderr)
            return 2
        runtime = build_runtime_config()
        sock = args.socket if args.socket is not None else runtime.serve_socket_path
        prov = args.provider if args.target == "subagent" else None
        if (
            args.target == "subagent"
            and prov
            and prov.strip().lower() != "fake"
        ):
            bad = _exit_if_static_llm_invalid(runtime, prov.strip().lower())
            if bad is not None:
                return bad
        try:
            asyncio.run(
                run_chat_client(
                    sock,
                    session_id=args.session,
                    target=args.target,
                    provider=prov,
                    verbose=args.verbose,
                )
            )
        except (OSError, RuntimeError) as exc:
            print(f"chief chat: {exc}", file=sys.stderr)
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
