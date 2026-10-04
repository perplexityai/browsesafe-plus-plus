"""Data contracts shared by the adaptive and browser modules (standard library only)."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclasses.dataclass(frozen=True)
class Slot:
    """Half-open character offsets in a single explicitly editable observation."""

    start: int
    end: int
    name: str = "page_insertion"


@dataclasses.dataclass(frozen=True)
class Surface:
    """An editable region in the context's single page observation."""

    surface_id: str
    slot: Slot
    description: str = ""


@dataclasses.dataclass(frozen=True)
class BrowserCheckpoint:
    """Adapter-owned snapshot immediately before a selected browser observation."""

    checkpoint_id: str
    tool_call_id: str
    tool_name: str
    url: str
    viewport: tuple[int, int] | None = None
    visible_history: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and x for x in
                   (self.checkpoint_id, self.tool_call_id, self.tool_name, self.url)):
            raise ValueError("Browser checkpoint requires snapshot, tool call, tool name and URL")
        if self.viewport is not None and (len(self.viewport) != 2 or any(
                type(x) is not int or x <= 0 for x in self.viewport)):
            raise ValueError("Viewport must contain positive width and height")
        json.dumps(self.visible_history, allow_nan=False)


@dataclasses.dataclass(frozen=True)
class Context:
    context_id: str
    task: str
    observation: str
    goal: str
    slot: Slot
    view: str = "markdown"
    source_revision: str | None = None
    source_entry_id: str | None = None
    source_family: str | None = None
    split: str = "development"
    attacker_context: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    surfaces: tuple[Surface, ...] = ()
    checkpoint: BrowserCheckpoint | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and x for x in
                   (self.context_id, self.task, self.observation, self.goal)):
            raise ValueError("Context identity, task, observation and goal are required")
        if self.view not in {"html", "markdown", "accessibility", "tool_response"}:
            raise ValueError("Unsupported observation view")
        if (type(self.slot.start) is not int or type(self.slot.end) is not int
                or not 0 <= self.slot.start <= self.slot.end <= len(self.observation)):
            raise ValueError("Slot must be inside the supplied observation")
        if not isinstance(self.attacker_context, Mapping):
            raise ValueError("attacker_context must be an explicitly supplied mapping")
        json.dumps(dict(self.attacker_context), allow_nan=False)
        ids: set[str] = set()
        for surface in self.surfaces:
            if (not isinstance(surface, Surface) or not surface.surface_id
                    or surface.surface_id in ids):
                raise ValueError("Surfaces require unique nonempty IDs")
            ids.add(surface.surface_id)
            slot = surface.slot
            if (type(slot.start) is not int or type(slot.end) is not int
                    or not 0 <= slot.start <= slot.end <= len(self.observation)):
                raise ValueError("Surface slot must be inside the observation")
        if self.checkpoint is not None and not isinstance(self.checkpoint, BrowserCheckpoint):
            raise ValueError("checkpoint must be BrowserCheckpoint")

    def selected_slot(self, surface_id: str | None = None) -> Slot:
        if surface_id is None:
            return self.slot
        for surface in self.surfaces:
            if surface.surface_id == surface_id:
                return surface.slot
        raise ValueError("Unknown surface ID")

    def insert(self, payload: str, surface_id: str | None = None) -> str:
        slot = self.selected_slot(surface_id)
        return self.observation[:slot.start] + payload + self.observation[slot.end:]


@dataclasses.dataclass(frozen=True)
class Seed:
    """An unchanged injection taken from the dataset, used as a starting candidate.

    ``attack_identified_ts`` is the dataset's detection time for that injection.
    """

    seed_id: str
    payload: str
    source_revision: str
    source_entry_id: str
    attack_identified_ts: str | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and x for x in
                   (self.seed_id, self.payload, self.source_revision, self.source_entry_id)):
            raise ValueError("Seed requires an ID, payload, source revision and source entry")


@dataclasses.dataclass(frozen=True)
class Usage:
    """Reported provider usage; unknown tokens or prices stay None."""

    model_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    lower_bounds: Mapping[str, int | float] = dataclasses.field(default_factory=dict)
    missing_reports: Mapping[str, int] = dataclasses.field(default_factory=dict)
    cost_basis: str | None = None

    def __post_init__(self) -> None:
        if type(self.model_calls) is not int or self.model_calls < 0:
            raise ValueError("model_calls must be a nonnegative integer")
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token usage must be nonnegative integers or None")
        if self.cost_usd is not None and (type(self.cost_usd) not in {int, float}
                or not math.isfinite(self.cost_usd) or self.cost_usd < 0):
            raise ValueError("Cost must be finite and nonnegative or None")
        fields = {"model_calls", "input_tokens", "output_tokens", "cost_usd"}
        for name, value in self.lower_bounds.items():
            if (name not in fields or getattr(self, name) is not None
                    or type(value) not in {int, float} or not math.isfinite(value) or value < 0
                    or (name != "cost_usd" and type(value) is not int)):
                raise ValueError("Lower bounds require an unknown dimension and nonnegative usage")
        for name, value in self.missing_reports.items():
            if name not in fields or getattr(self, name) is not None or type(value) is not int or value < 1:
                raise ValueError("Missing reports require an unknown dimension and positive count")
        if self.cost_basis not in {None, "provider_reported", "configured_token_rates", "mixed", "local"}:
            raise ValueError("Unknown cost basis")

    @classmethod
    def local(cls) -> Usage:
        """Explicitly report an operation that makes no model requests."""
        return cls(0, 0, 0, 0.0, cost_basis="local")

    @classmethod
    def aggregate(cls, values: Sequence[Usage]) -> Usage:
        totals, bounds, missing = {}, {}, {}
        for name in ("model_calls", "input_tokens", "output_tokens", "cost_usd"):
            known = sum(getattr(item, name) if getattr(item, name) is not None
                        else item.lower_bounds.get(name, 0) for item in values)
            count = sum(item.missing_reports.get(name, 1) for item in values
                        if getattr(item, name) is None)
            totals[name] = None if count else known
            if count:
                bounds[name], missing[name] = known, count
        bases = {item.cost_basis for item in values if item.cost_basis not in {None, "local"}}
        basis = next(iter(bases)) if len(bases) == 1 else "mixed" if bases else "local" if not values or all(item.cost_basis == "local" for item in values) else None
        return cls(**totals, lower_bounds=bounds, missing_reports=missing, cost_basis=basis)


def local_callback(callback):
    """Declare an async callback that never dispatches model requests."""
    callback.__browsesafe_local__ = True
    return callback


@dataclasses.dataclass(frozen=True)
class Candidate:
    payload: str
    strategy: str = "caller_proposed"
    parent_attempt_id: str | None = None
    source_seed_id: str | None = None
    surface_id: str | None = None
    derived_from_seed_id: str | None = None
    usage: Usage | None = None


@dataclasses.dataclass(frozen=True)
class TargetInput:
    """No attacker goal, labels, seed identity or evaluator annotations."""

    task: str
    observation: str
    view: str
    trust: str = "untrusted_observation"
    checkpoint: BrowserCheckpoint | None = None
    surface_id: str | None = None
    trial_seed: int | None = None
    max_model_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None


@dataclasses.dataclass(frozen=True)
class TargetResult:
    output: str
    events: tuple[Mapping[str, Any], ...] = ()
    detected: bool | None = None
    injection_observed: bool | None = None
    detector_score: float | None = None
    usage: Usage | None = None


class Target(Protocol):
    async def run(self, request: TargetInput) -> TargetResult: ...

    async def aclose(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class Trial:
    context: Context
    candidate: Candidate | None
    result: TargetResult
    phase: str


@dataclasses.dataclass(frozen=True)
class Verdict:
    task_success: bool | None
    attack_success: bool | None
    attack_valid: bool | None = None
    evidence: str = ""
    progress: float | None = None
    usage: Usage | None = None


@dataclasses.dataclass(frozen=True)
class Feedback:
    attempt_id: str
    candidate: Candidate | None
    status: str
    task_success: bool | None = None
    attack_success: bool | None = None
    attack_valid: bool | None = None
    detected: bool | None = None
    injection_observed: bool | None = None
    target_output: str | None = None
    confirmation_passed: bool | None = None
    progress: float | None = None
    events: tuple[Mapping[str, Any], ...] = ()
    judge_evidence: str | None = None
    detector_score: float | None = None


@dataclasses.dataclass(frozen=True)
class Budget:
    max_candidates: int = 8
    max_attacker_calls: int = 8
    max_target_calls: int = 10
    max_judge_calls: int = 10
    max_seconds: float = 600
    call_timeout: float = 60
    cleanup_timeout: float = 5
    max_payload_chars: int = 20_000
    max_model_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_candidates", "max_attacker_calls", "max_target_calls",
                     "max_judge_calls", "max_payload_chars"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_seconds", "call_timeout", "cleanup_timeout"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_model_calls", "max_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_cost_usd is not None and (type(self.max_cost_usd) not in {int, float}
                or not math.isfinite(self.max_cost_usd) or self.max_cost_usd <= 0):
            raise ValueError("max_cost_usd must be positive and finite or None")


@dataclasses.dataclass(frozen=True)
class RemainingBudget:
    attacker_calls: int
    target_calls: int
    judge_calls: int
    seconds: float


@dataclasses.dataclass
class CampaignResult:
    output_dir: Path
    summary: dict[str, Any]
    history: tuple[Feedback, ...]


@dataclasses.dataclass(frozen=True)
class Observation:
    task: str
    url: str
    view: str
    content: str
    screenshot_base64: str | None
    history: tuple[Mapping[str, Any], ...]
    viewport: tuple[int, int]
    step: int
    trial_seed: int | None = None
    remaining_tokens: int | None = None
    remaining_cost_usd: float | None = None


@dataclasses.dataclass(frozen=True)
class Action:
    """A single browser action from a model callback, with actual provider usage."""

    name: str
    arguments: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    usage: Usage | None = None

    def __post_init__(self) -> None:
        if self.name not in {"click", "fill", "type", "press", "scroll", "goto", "back", "finish"}:
            raise ValueError("Unknown browser action")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("Action arguments must be a mapping")
        json.dumps(dict(self.arguments), allow_nan=False)
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("Action usage must be Usage")

__all__ = ['Slot', 'Surface', 'BrowserCheckpoint', 'Context', 'Seed', 'Usage', 'Candidate', 'TargetInput', 'TargetResult', 'Target', 'Trial', 'Verdict', 'Feedback', 'Budget', 'RemainingBudget', 'CampaignResult', 'Observation', 'Action', 'local_callback']
