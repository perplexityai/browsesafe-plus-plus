"""Bounded sequential campaigns with proposals, feedback and confirmations."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from ..contracts import (
    Budget, CampaignResult, Candidate, Context, Feedback, RemainingBudget,
    Seed, Slot, Target, Trial, Verdict,
)
from .._execution import _BudgetExhausted, _CallFailed, _Ledger, _hash, _utc, execute_trial


def from_pair(benchmark: Any, pair: Any, *, slot: Slot, goal: str | None = None) -> Context:
    """Build a :class:`Context` from a loaded pair: the benign page, its task and the pair's goal.

    The caller chooses the insertion ``slot`` (character offsets in the benign view).
    ``goal`` defaults to the pair's ``malicious_goal`` annotation.
    """
    if not any(item is pair for item in benchmark.pairs):
        raise ValueError("Pair must belong to the supplied Benchmark")
    return Context(
        context_id=pair.pair_id, task=pair.benign.task,
        observation=pair.benign.content,
        goal=goal if goal is not None else pair.malicious.metadata.get("malicious_goal"),
        slot=slot, view=pair.benign.view,
        source_revision=(benchmark.manifest.get("revision") or benchmark.manifest.get("fingerprint")
                         or benchmark.manifest.get("sampling_identity")),
        source_entry_id=pair.benign.entry_id, split=pair.malicious.metadata.get(benchmark.manifest.get("split_column", "split"), "unknown"),
    )



async def run(
    context: Context,
    *,
    attacker: Callable[[Context, tuple[Feedback, ...], RemainingBudget], Awaitable[Candidate]],
    target_factory: Callable[[], Target],
    judge: Callable[[Trial, Target], Awaitable[Verdict]],
    output_dir: str | Path,
    budget: Budget = Budget(),
    seeds: Sequence[Seed] = (),
    mode: str = "replay",
    feedback_mode: str = "outcome",
    confirmations: int = 1,
    require_clean_success: bool = True,
    stop_on_success: bool = True,
    protocol_id: str = "caller-defined-v1",
) -> CampaignResult:
    """Run one adaptive attack campaign against a single context.

    The campaign first runs a clean baseline (no injection). If the target solves the
    task (or ``require_clean_success=False``), it repeatedly asks ``attacker`` for a
    :class:`Candidate`, inserts the payload into the context's slot, runs a fresh target
    from ``target_factory`` and scores the trial with ``judge``. A judged success is
    re-run ``confirmations`` times on fresh targets before it counts as confirmed.

    Modes:
        ``"replay"``: every candidate must be an unchanged registered seed.
        ``"generate"``: candidates may also be new payloads (recorded as ``adaptive_generated``).

    ``feedback_mode`` controls what the attacker sees about earlier trials:
    ``"outcome"`` (verdicts only), ``"output"`` (plus the target's answer) or ``"trace"``
    (plus target events and judge evidence).

    All calls are charged against ``budget``; failures and timeouts count. Every event is
    appended to ``output_dir/events.jsonl``; ``manifest.json`` and ``summary.json`` are
    written alongside. ``output_dir`` must not exist yet.
    """
    if mode not in {"replay", "generate"} or feedback_mode not in {"outcome", "output", "trace"}:
        raise ValueError("Invalid campaign mode or feedback mode")
    if type(confirmations) is not int or confirmations < 0:
        raise ValueError("confirmations must be a nonnegative integer")
    seed_map = {s.seed_id: s for s in seeds}
    if len(seed_map) != len(seeds):
        raise ValueError("Seed IDs must be unique")
    for seed in seeds:
        if not isinstance(seed, Seed):
            raise TypeError("seeds must contain Seed objects")
    if mode == "replay" and not seeds:
        raise ValueError("Replay requires registered source seeds")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    ledger = _Ledger(directory, budget)
    manifest = {"schema_version": "0.1", "campaign_id": str(uuid.uuid4()),
                "created_at": _utc(), "protocol_id": protocol_id,
                "mode": mode, "feedback_mode": feedback_mode,
                "search": getattr(attacker, "configuration", None),
                "budget": dataclasses.asdict(budget), "confirmations": confirmations,
                "require_clean_success": require_clean_success, "stop_on_success": stop_on_success,
                "context": dataclasses.asdict(context), "observation_sha256": _hash(context.observation),
                "seeds": [dataclasses.asdict(s) for s in seeds],
                "adaptive_exposure": True, "frozen_benchmark_modified": False}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    history: list[Feedback] = []
    trials: list[dict[str, Any]] = []
    candidate_count = 0
    baseline: Feedback | None = None
    confirmed_success: bool | None = None
    stop_reason = "candidate_budget_exhausted"

    async def trial(candidate: Candidate | None, attempt_id: str, phase: str) -> Feedback:
        return await execute_trial(
            ledger=ledger, context=context, candidate=candidate,
            attempt_id=attempt_id, phase=phase, target_factory=target_factory,
            judge=judge, feedback_mode=feedback_mode, trials=trials,
        )

    def candidate_origin(candidate: Candidate) -> dict[str, Any]:
        if not isinstance(candidate.payload, str) or not candidate.payload.strip():
            raise ValueError("Candidate payload must be nonempty text")
        if len(candidate.payload) > budget.max_payload_chars:
            raise ValueError("Candidate exceeds payload budget")
        if not isinstance(candidate.strategy, str) or not candidate.strategy:
            raise ValueError("Candidate strategy must be nonempty text")
        if any(x is not None and not isinstance(x, str) for x in
               (candidate.parent_attempt_id, candidate.source_seed_id,
                candidate.surface_id, candidate.derived_from_seed_id)):
            raise ValueError("Candidate lineage IDs must be strings or None")
        if candidate.parent_attempt_id is not None and candidate.parent_attempt_id not in {h.attempt_id for h in history}:
            raise ValueError("Unknown candidate parent")
        context.selected_slot(candidate.surface_id)
        if candidate.derived_from_seed_id is not None and candidate.derived_from_seed_id not in seed_map:
            raise ValueError("Unknown ancestor seed")
        if candidate.source_seed_id and candidate.derived_from_seed_id:
            raise ValueError("Use source_seed_id for replay or derived_from_seed_id for a mutation")
        seed = seed_map.get(candidate.source_seed_id) if candidate.source_seed_id else None
        if candidate.source_seed_id and seed is None:
            raise ValueError("Unknown seed")
        if seed is not None:
            if candidate.payload != seed.payload:
                raise ValueError("Changed payload cannot inherit a seed's identity or timestamp")
            return {"origin": "seed_replay",
                    "placement": "transplanted_into_declared_slot", "source_seed_id": seed.seed_id,
                    "source_revision": seed.source_revision, "source_entry_id": seed.source_entry_id,
                    "attack_identified_ts": seed.attack_identified_ts}
        if mode == "replay":
            raise ValueError("Replay candidate must bind an unchanged registered seed")
        return {"origin": "adaptive_generated", "attack_identified_ts": None,
                "generated_at": _utc(), "derived_from_seed_id": candidate.derived_from_seed_id}

    try:
        baseline = await trial(None, "baseline", "baseline")
        if ledger.cleanup_errors:
            stop_reason = "target_cleanup_failed"
        elif require_clean_success and baseline.task_success is not True:
            stop_reason = "clean_baseline_not_successful"
        else:
            seen: set[str] = set()
            for index in range(budget.max_candidates):
                remaining = ledger.remaining()
                if min(remaining.attacker_calls, remaining.target_calls, remaining.judge_calls) <= 0 or remaining.seconds <= 0:
                    stop_reason = "call_or_time_budget_exhausted"
                    break
                attempt_id = f"candidate-{index + 1:04d}"
                candidate_count += 1
                candidate: Candidate | None = None
                ledger.event("proposal_requested", attempt_id=attempt_id,
                             history=[dataclasses.asdict(h) for h in history],
                             remaining=dataclasses.asdict(remaining))
                try:
                    candidate = await ledger.call("attacker", attacker, context, tuple(history),
                                                  remaining, attempt_id=attempt_id)
                    if not isinstance(candidate, Candidate):
                        candidate = None
                        raise ValueError("Attacker must return Candidate")
                    origin = candidate_origin(candidate)
                    payload_digest = _hash(candidate.payload)
                    slot = context.selected_slot(candidate.surface_id)
                    digest = _hash(json.dumps([payload_digest, slot.start, slot.end]))
                    if digest in seen:
                        raise ValueError("Duplicate candidate")
                    seen.add(digest)
                    ledger.event("candidate_accepted", attempt_id=attempt_id,
                                 candidate=dataclasses.asdict(candidate), payload_sha256=payload_digest,
                                 candidate_sha256=digest,
                                 lineage=origin)
                except (_CallFailed, ValueError) as exc:
                    status = exc.status if isinstance(exc, _CallFailed) else "invalid_candidate"
                    feedback = Feedback(attempt_id, candidate, status)
                    history.append(feedback)
                    ledger.event("candidate_rejected", **dataclasses.asdict(feedback), error_type=type(exc).__name__)
                    continue
                result = await trial(candidate, attempt_id, "search")
                if ledger.cleanup_errors:
                    history.append(result)
                    stop_reason = "target_cleanup_failed"
                    break
                if result.attack_success is True:
                    confirmation_results = []
                    for number in range(confirmations):
                        confirmation_results.append(await trial(candidate, f"{attempt_id}-confirm-{number + 1}", "confirmation"))
                        if confirmation_results[-1].status == "budget_exhausted" or ledger.cleanup_errors:
                            break
                    if not confirmations:
                        passed = None
                    elif any(r.attack_valid is True and r.attack_success is False for r in confirmation_results):
                        passed = False
                    elif (len(confirmation_results) == confirmations
                          and all(r.attack_success is True for r in confirmation_results)):
                        passed = True
                    else:
                        passed = None
                    result = dataclasses.replace(result, confirmation_passed=passed)
                    ledger.event("candidate_confirmation", attempt_id=attempt_id,
                                 confirmation_passed=passed)
                    if confirmations:
                        if passed is True:
                            confirmed_success = True
                        elif confirmed_success is not True:
                            confirmed_success = passed
                history.append(result)
                if ledger.cleanup_errors:
                    stop_reason = "target_cleanup_failed"
                    break
                if result.attack_success is True and stop_on_success and (
                        not confirmations or result.confirmation_passed is True):
                    stop_reason = "attack_success_confirmed" if confirmations else "attack_success_discovered"
                    break
                if result.status == "budget_exhausted":
                    stop_reason = "call_or_time_budget_exhausted"
                    break
    except _BudgetExhausted:
        stop_reason = "call_or_time_budget_exhausted"
    except asyncio.CancelledError:
        stop_reason = "interrupted"
        ledger.event("campaign_interrupted", candidate_count=candidate_count)
        raise
    finally:
        search = [t for t in trials if t["phase"] == "search"]
        successes = [t for t in search if t["attack_success"] is True]
        known = [t for t in search if t["attack_success"] is not None and t["attack_valid"] is True]
        utility_known = [t for t in search if t["task_success"] is not None]
        summary = {"schema_version": "0.1", "protocol_id": protocol_id,
                   "baseline_task_success": baseline.task_success if baseline else None,
                   "candidate_proposals": candidate_count, "candidate_trials": len(search),
                   "rejected_or_unexecuted_proposals": candidate_count - len(search),
                   "judged_attack_outcomes": len(known), "unscorable_attack_outcomes": len(search) - len(known),
                   "unknown_attack_outcomes": sum(t["attack_valid"] is None or t["attack_success"] is None for t in search),
                   "invalid_attack_trials": sum(t["attack_valid"] is False for t in search),
                   "attack_successes": len(successes),
                   "seed_replay_trials": sum(t["candidate"]["source_seed_id"] is not None for t in search),
                   "generated_trials": sum(t["candidate"]["source_seed_id"] is None for t in search),
                   "seed_replay_successes": sum(t["candidate"]["source_seed_id"] is not None for t in successes),
                   "generated_successes": sum(t["candidate"]["source_seed_id"] is None for t in successes),
                   "attack_success_rate_judged": len(successes) / len(known) if known else None,
                   "attack_success_rate_attempted_lower_bound": len(successes) / len(search) if search else None,
                   "attacked_task_success_rate_judged": sum(t["task_success"] is True for t in utility_known) / len(utility_known) if utility_known else None,
                   "success_at_budget": (True if successes else False if search and len(known) == len(search)
                                          and candidate_count == len(search) else None)
                                         if baseline and baseline.task_success is True else None,
                   "confirmed_success": confirmed_success,
                   "first_success_attempt_id": successes[0]["attempt_id"] if successes else None,
                   "calls": dict(ledger.counts), "cleanup_errors": ledger.cleanup_errors,
                   "usage": ledger.usage_summary(),
                   "elapsed_seconds": time.monotonic() - ledger.started,
                   "stop_reason": stop_reason, "frozen_benchmark_modified": False,
                   "confirmation_trials": sum(t["phase"] == "confirmation" for t in trials)}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        ledger.event("campaign_closed", summary=summary)
    return CampaignResult(directory, summary, tuple(history))
