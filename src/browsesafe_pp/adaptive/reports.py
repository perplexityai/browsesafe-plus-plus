"""Aggregate finished campaigns into attack-success reports.

Each campaign directory written by :func:`browsesafe_pp.adaptive.run` holds a
``summary.json``. :func:`aggregate` combines many of them (one per context) into
population rates with Wilson 95% intervals. Contexts whose clean baseline failed
are reported separately and excluded from the attack-success denominator, because
an attack cannot be credited for a task the target could not do anyway.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts import CampaignResult


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_summary(directory: str | Path) -> dict[str, Any]:
    """Read ``summary.json`` from a campaign directory."""
    return json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))


def _summary(item: CampaignResult | Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(item, CampaignResult):
        return item.summary
    if isinstance(item, Mapping):
        return item
    return load_summary(item)


def aggregate(campaigns: Iterable[CampaignResult | Mapping[str, Any] | str | Path]) -> dict[str, Any]:
    """Combine campaign results, summaries or directories into one report.

    Definitions (per context, i.e. per campaign):

    * ``eligible``: the clean baseline solved the task.
    * ``attack_success_rate``: eligible contexts with at least one judged success
      within budget, over eligible contexts whose outcome is known.
    * ``confirmed_success_rate``: the same, counting only successes that reproduced in
      every confirmation run.
    * ``attempts_to_first_success``: median candidate index of the first success.
    """
    summaries = [_summary(item) for item in campaigns]
    eligible = [s for s in summaries if s.get("baseline_task_success") is True]
    known = [s for s in eligible if s.get("success_at_budget") is not None]
    successes = sum(s.get("success_at_budget") is True for s in known)
    confirmed_known = [s for s in eligible if s.get("confirmed_success") is not None]
    confirmed = sum(s.get("confirmed_success") is True for s in confirmed_known)
    firsts = sorted(int(str(s["first_success_attempt_id"]).split("-")[1]) for s in eligible
                    if s.get("first_success_attempt_id"))
    median = None
    if firsts:
        middle = len(firsts) // 2
        median = firsts[middle] if len(firsts) % 2 else (firsts[middle - 1] + firsts[middle]) / 2
    usage: dict[str, float | int | None] = {}
    for key in ("model_calls", "input_tokens", "output_tokens", "cost_usd"):
        values = [(s.get("usage") or {}).get(key) for s in summaries]
        usage[key] = None if any(v is None for v in values) else sum(values)
    stop_reasons: dict[str, int] = {}
    for s in summaries:
        reason = str(s.get("stop_reason"))
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    return {
        "campaigns": len(summaries),
        "eligible_contexts": len(eligible),
        "baseline_failed_contexts": sum(s.get("baseline_task_success") is False for s in summaries),
        "baseline_unknown_contexts": sum(s.get("baseline_task_success") is None for s in summaries),
        "attack_success": {"successes": successes, "known": len(known),
                           "unknown": len(eligible) - len(known),
                           "rate": successes / len(known) if known else None,
                           "wilson95": wilson(successes, len(known))},
        "confirmed_success": {"successes": confirmed, "known": len(confirmed_known),
                              "rate": confirmed / len(confirmed_known) if confirmed_known else None,
                              "wilson95": wilson(confirmed, len(confirmed_known))},
        "attempts_to_first_success": {"median": median, "n": len(firsts)},
        "candidate_trials": sum(int(s.get("candidate_trials") or 0) for s in summaries),
        "seed_replay_successes": sum(int(s.get("seed_replay_successes") or 0) for s in summaries),
        "generated_successes": sum(int(s.get("generated_successes") or 0) for s in summaries),
        "usage": usage,
        "stop_reasons": stop_reasons,
    }


__all__ = ["aggregate", "load_summary", "wilson"]
