"""Command-line interface for running chief episodes.

The ``chief`` console script dispatches subcommands (currently ``run``).
"""

from __future__ import annotations

import argparse
import json

from chief.brain import Brain, FakeBrain
from chief.engine import run_episode
from chief.llm import HttpChatCompletionsBrain
from chief.tools import build_registry


def _select_brain(name: str) -> Brain:
    """Instantiate the planner implementation selected on the CLI.

    Args:
        name: ``fake`` scripted planner, or ``llm`` for any OpenAI-compatible
            ``/v1/chat/completions`` endpoint (configure via ``CHIEF_LLM_*``).

    Returns:
        Concrete :class:`~chief.brain.Brain`.
    """
    key = name.strip().lower()
    if key == "fake":
        return FakeBrain()
    return HttpChatCompletionsBrain.from_env()


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
        choices=("fake", "llm"),
        default="fake",
        help="Planner: fake (offline) or llm (OpenAI-compatible Chat Completions API)",
    )
    run_p.add_argument(
        "--max-cycles",
        type=int,
        default=16,
        help="Hard cap on replan cycles (default 16)",
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        task = " ".join(args.task).strip()
        brain = _select_brain(args.brain)
        episode = run_episode(
            task,
            brain=brain,
            tools=build_registry(),
            max_cycles=args.max_cycles,
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
