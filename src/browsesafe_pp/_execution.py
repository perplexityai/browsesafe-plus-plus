"""Shared dispatch, usage accounting and isolated trial execution."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import inspect
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from .contracts import (
    Budget, Candidate, Context, Feedback, RemainingBudget, Target, TargetInput,
    TargetResult, Trial, Usage, Verdict,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class _BudgetExhausted(Exception):
    pass


class _CallFailed(Exception):
    def __init__(self, status: str) -> None:
        self.status = status


class _Ledger:
    def __init__(self, directory: Path, budget: Budget) -> None:
        self.directory = directory
        self.budget = budget
        self.started = time.monotonic()
        self.counts = {"attacker": 0, "target": 0, "judge": 0}
        self.sequence = 0
        self.cleanup_errors = 0
        self.usage_known = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        self.usage_missing = dict.fromkeys(self.usage_known, 0)
        self.cost_bases: set[str] = set()

    def account(self, usage: Usage | None) -> None:
        if usage is not None and not isinstance(usage, Usage):
            raise ValueError("Callbacks must report Usage or None")
        for name in self.usage_known:
            value = getattr(usage, name) if usage is not None else None
            if value is None:
                self.usage_missing[name] += usage.missing_reports.get(name, 1) if usage else 1
                self.usage_known[name] += usage.lower_bounds.get(name, 0) if usage else 0
            else:
                self.usage_known[name] += value
        if usage and usage.cost_basis:
            self.cost_bases.add(usage.cost_basis)

    def usage_summary(self) -> dict[str, Any]:
        return {**{key: value if not self.usage_missing[key] else None
                   for key, value in self.usage_known.items()},
                "reported_lower_bounds": dict(self.usage_known),
                "unreported_callbacks": dict(self.usage_missing),
                "cost_bases": sorted(self.cost_bases),
                "overshoot": {name: max(0, sum(self.usage_known[k] for k in keys) - cap)
                              for name, keys, cap in (
                                  ("model_calls", ("model_calls",), self.budget.max_model_calls),
                                  ("tokens", ("input_tokens", "output_tokens"), self.budget.max_tokens),
                                  ("cost_usd", ("cost_usd",), self.budget.max_cost_usd)) if cap is not None}}

    def usage_limits(self) -> dict[str, Any]:
        limits = {}
        for name, keys in (("max_model_calls", ("model_calls",)),
                           ("max_tokens", ("input_tokens", "output_tokens")),
                           ("max_cost_usd", ("cost_usd",))):
            cap = getattr(self.budget, name)
            if cap is None:
                limits[name] = None
                continue
            if any(self.usage_missing[key] for key in keys):
                raise _BudgetExhausted("unreported_usage")
            left = cap - sum(self.usage_known[key] for key in keys)
            if left <= 0:
                raise _BudgetExhausted(name)
            limits[name] = left
        return limits

    def event(self, event: str, **fields: Any) -> None:
        self.sequence += 1
        record = {"sequence": self.sequence, "utc": _utc(), "event": event,
                  "calls": dict(self.counts), **fields}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def remaining(self) -> RemainingBudget:
        return RemainingBudget(
            self.budget.max_attacker_calls - self.counts["attacker"],
            self.budget.max_target_calls - self.counts["target"],
            self.budget.max_judge_calls - self.counts["judge"],
            max(0, self.budget.max_seconds - (time.monotonic() - self.started)),
        )

    async def call(self, kind: str, callback: Callable[..., Any], *args: Any,
                   attempt_id: str) -> Any:
        remaining = self.remaining()
        is_local = getattr(callback, "__browsesafe_local__", False)
        if not is_local:
            self.usage_limits()
        if remaining.seconds <= 0 or self.counts[kind] >= getattr(self.budget, f"max_{kind}_calls"):
            raise _BudgetExhausted(kind)
        # Reserve and persist before dispatch, including failures and interrupted calls.
        self.counts[kind] += 1
        self.event("call_started", kind=kind, attempt_id=attempt_id)
        accounted = False

        def reported_usage(value):
            usage = getattr(value, "usage", None)
            if usage is None and is_local:
                return Usage.local()
            if usage is not None:
                return usage
            snapshot = getattr(getattr(callback, "__self__", None), "usage_snapshot", None)
            return snapshot() if snapshot else None

        try:
            if not (inspect.iscoroutinefunction(callback)
                    or inspect.iscoroutinefunction(getattr(callback, "__call__", None))):
                raise TypeError("Execution callbacks must be async")
            answer = await asyncio.wait_for(callback(*args),
                                             timeout=min(self.budget.call_timeout, remaining.seconds))
            usage = reported_usage(answer)
            self.account(usage)
            accounted = True
            if is_local and usage is not None and usage.model_calls:
                raise ValueError("A local callback reported model requests")
            # Serialization is checked before the response can affect a subsequent call.
            encoded = dataclasses.asdict(answer) if dataclasses.is_dataclass(answer) else answer
            json.dumps(encoded, allow_nan=False)
            self.event("call_completed", kind=kind, attempt_id=attempt_id, response=encoded)
            return answer
        except asyncio.CancelledError as exc:
            if not accounted:
                self.account(reported_usage(exc))
            self.event("call_interrupted", kind=kind, attempt_id=attempt_id)
            raise
        except Exception as exc:
            # A failed provider request may already have incurred usage.
            if not accounted:
                usage = reported_usage(exc)
                self.account(usage if isinstance(usage, Usage) else None)
            status = "timeout" if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else "error"
            self.event("call_failed", kind=kind, attempt_id=attempt_id,
                       status=status, error_type=type(exc).__name__)
            raise _CallFailed(status) from None


def _check_verdict(verdict: Any, phase: str) -> Verdict:
    if not isinstance(verdict, Verdict):
        raise ValueError("Judge must return Verdict")
    if any(x is not None and type(x) is not bool for x in
           (verdict.task_success, verdict.attack_success, verdict.attack_valid)):
        raise ValueError("Verdict decisions must be bool or None")
    if not isinstance(verdict.evidence, str):
        raise ValueError("Verdict evidence must be a string")
    if verdict.progress is not None and (type(verdict.progress) not in {int, float}
            or not math.isfinite(verdict.progress) or not 0 <= verdict.progress <= 1):
        raise ValueError("Progress must be None or a finite score between zero and one")
    if verdict.attack_success is True and (phase == "baseline"
            or verdict.attack_valid is not True or not verdict.evidence.strip()):
        raise ValueError("Attack success requires a valid attack and explicit evidence")
    return verdict



async def execute_trial(
    *, ledger: _Ledger, context: Context, candidate: Candidate | None,
    attempt_id: str, phase: str, target_factory: Callable[[], Target],
    judge: Callable[[Trial, Target], Awaitable[Verdict]],
    feedback_mode: str, trials: list[dict[str, Any]],
    trial_seed: int | None = None,
) -> Feedback:
    """Execute and record one isolated target/judge trial, including cleanup."""
    target: Target | None = None
    observed = context.observation if candidate is None else context.insert(candidate.payload, candidate.surface_id)
    ledger.event("trial_started", attempt_id=attempt_id, phase=phase,
                 observation_sha256=_hash(observed))
    try:
        if ledger.remaining().seconds <= 0:
            raise _BudgetExhausted("wall_time")
        # Factory allocates isolated local state; network/model work belongs in run().
        target = target_factory()
        if inspect.isawaitable(target):
            raise TypeError("Factory must synchronously return an async Target")
        response = await ledger.call("target", target.run,
                                    TargetInput(context.task, observed, context.view,
                                                checkpoint=context.checkpoint,
                                                surface_id=candidate.surface_id if candidate else None,
                                                trial_seed=trial_seed,
                                                **ledger.usage_limits()),
                                    attempt_id=attempt_id)
        if not isinstance(response, TargetResult) or not isinstance(response.output, str):
            raise ValueError("Target must return TargetResult")
        for decision in (response.detected, response.injection_observed):
            if decision is not None and type(decision) is not bool:
                raise ValueError("Target observability fields must be bool or None")
        if response.detector_score is not None and (type(response.detector_score) not in {int, float}
                or not math.isfinite(response.detector_score) or not 0 <= response.detector_score <= 1):
            raise ValueError("Detector score must be None or between zero and one")
        if not isinstance(response.events, tuple) or any(not isinstance(e, Mapping) for e in response.events):
            raise ValueError("Target events must be a tuple of mappings")
        verdict = _check_verdict(await ledger.call(
            "judge", judge, Trial(context, candidate, response, phase), target,
            attempt_id=attempt_id), phase)
        feedback = Feedback(attempt_id, candidate, "judged", verdict.task_success,
                            verdict.attack_success, verdict.attack_valid, response.detected,
                            response.injection_observed,
                            response.output if feedback_mode in {"output", "trace"} else None,
                            progress=verdict.progress,
                            events=response.events if feedback_mode == "trace" else (),
                            judge_evidence=verdict.evidence if feedback_mode == "trace" else None,
                            detector_score=response.detector_score)
    except _CallFailed as exc:
        feedback = Feedback(attempt_id, candidate, exc.status)
    except _BudgetExhausted:
        feedback = Feedback(attempt_id, candidate, "budget_exhausted")
    except asyncio.CancelledError:
        trials.append({"phase": phase, **dataclasses.asdict(
            Feedback(attempt_id, candidate, "interrupted"))})
        ledger.event("trial_interrupted", attempt_id=attempt_id, phase=phase)
        raise
    except Exception as exc:
        feedback = Feedback(attempt_id, candidate, "invalid_result")
        ledger.event("trial_invalid", attempt_id=attempt_id, error_type=type(exc).__name__)
    finally:
        if target is not None:
            try:
                await asyncio.wait_for(target.aclose(), timeout=ledger.budget.cleanup_timeout)
            except Exception as exc:
                ledger.cleanup_errors += 1
                ledger.event("cleanup_failed", attempt_id=attempt_id, error_type=type(exc).__name__)
    record = {"phase": phase, **dataclasses.asdict(feedback)}
    trials.append(record)
    ledger.event("trial_completed", **record)
    return feedback
