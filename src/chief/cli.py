"""Command-line interface for running chief episodes.

The ``chief`` console script dispatches subcommands (currently ``run``).
"""

from __future__ import annotations

import argparse
import asyncio
import json

from chief.brain import Brain, FakeBrain
from chief.engine import run_episode
from chief.llm import CustomChatCompletionsBrain, OpenAiChatCompletionsBrain
from chief.config import RuntimeConfig, build_runtime_config
from chief.tools import build_registry


def _select_brain(name: str, runtime: RuntimeConfig) -> Brain:
    """Instantiate the planner implementation selected on the CLI.

    Args:
        name: ``fake`` scripted planner, ``llm`` for **custom** OpenAI-compat ``/v1`` gateway
            (``[custom_llm]`` / ``CHIEF_LLM_*``), or ``openai`` for vendor OpenAI (``[openai]`` / ``CHIEF_OPENAI_*``).
        runtime: Configuration snapshot for this process.

    Returns:
        Concrete :class:`~chief.brain.Brain`.
    """
    key = name.strip().lower()
    if key == "fake":
        return FakeBrain(runtime)
    if key == "llm":
        return CustomChatCompletionsBrain.from_runtime(runtime)
    if key == "openai":
        return OpenAiChatCompletionsBrain.from_runtime(runtime)
    raise ValueError(f"unknown brain: {name!r}")


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested subcommand.

    Args:
        argv: Argument list; ``None`` lets :mod:`argparse` read the process argv.

    Returns:
        Shell exit code: ``0`` on completed episode, ``1`` on failure paths,
        ``2`` when the subcommand is unknown.
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
        "--brain",
        choices=("fake", "llm", "openai"),
        default="fake",
        help="Planner: fake | llm ([custom_llm] / CHIEF_LLM_*) | openai ([openai] / CHIEF_OPENAI_*)",
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

    args = parser.parse_args(argv)

    if args.cmd == "run":
        task = " ".join(args.task).strip()
        runtime = build_runtime_config()
        brain = _select_brain(args.brain, runtime)
        max_c = args.max_cycles if args.max_cycles is not None else runtime.episode_max_cycles
        episode = asyncio.run(
            run_episode(
                task,
                runtime=runtime,
                brain=brain,
                tools=build_registry(runtime),
                max_cycles=max_c,
            )
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "id": episode.id,
                        "status": episode.status.value,
                        "artifact": episode.artifact,
                        "ticks": len(episode.ticks),
                        "task": task,
                        "brain": args.brain,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"episode={episode.id} status={episode.status.value} ticks={len(episode.ticks)} brain={args.brain}")
            if episode.artifact:
                print(f"artifact={episode.artifact}")
            for t in episode.ticks:
                print(f"  [{t.index}] cycle={t.cycle} phase={t.phase.value} {t.data}")

        return 0 if episode.status.value == "completed" else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
