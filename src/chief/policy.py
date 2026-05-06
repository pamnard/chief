"""Policy gate evaluated after Reason and before Act (v0 stub).

Extensible hook for quotas, allowlists, and budget enforcement.
"""

from __future__ import annotations

from chief.domain import FinalIntent, Intent, PolicyResult, ToolIntent


def evaluate_intent(intent: Intent, cycle: int, max_cycles: int) -> PolicyResult:
    """Decide whether an intent may execute in the current cycle.

    Args:
        intent: Planner output (tool call or final message).
        cycle: Current orchestrator cycle index (0-based).
        max_cycles: Maximum allowed cycles before hard stop.

    Returns:
        ``PolicyResult`` with ``allowed`` False when budget is exhausted or intent
        is unknown to this stub implementation.
    """
    if cycle >= max_cycles:
        return PolicyResult(False, "max_cycles_exceeded")
    if isinstance(intent, ToolIntent) and intent.tool == "noop":
        return PolicyResult(True)
    if isinstance(intent, ToolIntent) and intent.tool in ("echo", "broken"):
        return PolicyResult(True)
    if isinstance(intent, FinalIntent):
        return PolicyResult(True)
    return PolicyResult(False, f"unknown_intent:{intent!r}")
