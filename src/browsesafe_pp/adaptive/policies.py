"""Diversified proposal selection for adaptive browser experiments."""
from __future__ import annotations

import dataclasses
import hashlib
import math
import random
from typing import Awaitable, Callable, Sequence

from ..contracts import Candidate, Context, Feedback, RemainingBudget, Usage


@dataclasses.dataclass(frozen=True)
class SearchRequest:
    context: Context
    parents: tuple[Feedback, ...]
    recent: tuple[Feedback, ...]
    remaining: RemainingBudget
    restart: bool
    seeds: tuple[Candidate, ...] = ()


def _rank(feedback: Feedback) -> tuple[int, float, int]:
    """Rank confirmed outcomes above intermediate progress and failed repeats."""
    progress = feedback.progress
    progress = float(progress) if (isinstance(progress, (int, float))
                                  and not isinstance(progress, bool)
                                  and math.isfinite(progress)
                                  and 0 <= progress <= 1) else 0.0
    observed = int(feedback.injection_observed is True)
    if feedback.confirmation_passed is False:
        return (0, 0.0, observed)
    if feedback.attack_valid is True and feedback.attack_success is True:
        return (4 if feedback.confirmation_passed is True else 3, progress, observed)
    return (2 if feedback.attack_valid is True else 1, progress, observed)


class PortfolioSearch:
    """Choose diverse parents, then invoke the supplied mutator exactly once.

    Each archive cell groups one surface, strategy and power-of-two payload
    length bucket. The highest-ranked trial survives in each cell. One elite is
    always selected; the remaining parents are sampled with rank weighting.
    Archives are rebuilt from the supplied campaign history on every call.
    Initial candidates are proposed once each before mutation. Their unchanged
    starting points remain available in ``SearchRequest.seeds``, including on
    restarts. Mutated payloads use parent lineage rather than inheriting an
    unchanged seed's ``source_seed_id``.
    The last eight records, including rejected proposals, remain in ``recent``.
    Every ``restart_every`` mutation proposals the mutator receives no parents.
    """

    def __init__(self, mutate: Callable[[SearchRequest], Awaitable[Candidate]], *,
                 seed: int = 42, parent_count: int = 8, restart_every: int = 8,
                 initial_candidates: tuple[Candidate, ...] = ()) -> None:
        if not callable(mutate):
            raise TypeError("mutate must be an asynchronous callable")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        for name, value in (("parent_count", parent_count), ("restart_every", restart_every)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(initial_candidates, tuple) or not all(
                isinstance(candidate, Candidate) for candidate in initial_candidates):
            raise TypeError("initial_candidates must be a tuple of Candidate objects")
        self.mutate = mutate
        self.seed = seed
        self.parent_count = parent_count
        self.restart_every = restart_every
        self.initial_candidates = initial_candidates

    @property
    def configuration(self) -> dict:
        return {"controller": "diverse_elite_archive", "seed": self.seed,
                "parent_count": self.parent_count, "restart_every": self.restart_every,
                "initial_candidates": [dataclasses.asdict(c) for c in self.initial_candidates]}

    async def __call__(self, context: Context, history: Sequence[Feedback],
                       remaining: RemainingBudget) -> Candidate:
        history = tuple(history)
        if len(history) < len(self.initial_candidates):
            return dataclasses.replace(self.initial_candidates[len(history)], usage=Usage.local())
        mutation_index = len(history) - len(self.initial_candidates)
        restart = not history or (mutation_index > 0 and mutation_index % self.restart_every == 0)
        parents: tuple[Feedback, ...] = ()
        if not restart:
            archive: dict[tuple[str | None, str, int], tuple[int, Feedback]] = {}
            for index, feedback in enumerate(history):
                candidate = feedback.candidate
                if (feedback.status != "judged" or candidate is None
                        or feedback.attack_valid is False):
                    continue
                cell = (candidate.surface_id, candidate.strategy,
                        max(1, len(candidate.payload)).bit_length() - 1)
                previous = archive.get(cell)
                if previous is None or (_rank(feedback), index) > (_rank(previous[1]), previous[0]):
                    archive[cell] = (index, feedback)
            pool = sorted(archive.values(), key=lambda item: (_rank(item[1]), item[0]), reverse=True)
            if pool:
                digest = hashlib.sha256(f"{self.seed}\0{context.context_id}\0".encode())
                for feedback in history:
                    digest.update(feedback.attempt_id.encode())
                    digest.update(b"\0")
                    if feedback.candidate is not None:
                        digest.update(hashlib.sha256(feedback.candidate.payload.encode()).digest())
                rng = random.Random(int.from_bytes(digest.digest(), "big"))
                selected = [pool.pop(0)]
                while pool and len(selected) < self.parent_count:
                    weights = [1 / (index + 1) for index in range(len(pool))]
                    chosen = rng.choices(range(len(pool)), weights=weights, k=1)[0]
                    selected.append(pool.pop(chosen))
                selected.sort(key=lambda item: (_rank(item[1]), item[0]), reverse=True)
                parents = tuple(feedback for _, feedback in selected)
            else:
                restart = True
        request = SearchRequest(context, parents, history[-8:], remaining, restart,
                                seeds=self.initial_candidates)
        candidate = await self.mutate(request)
        if not isinstance(candidate, Candidate):
            raise TypeError("mutate must return Candidate")
        return candidate


@dataclasses.dataclass(frozen=True)
class Arm:
    """One available evaluation choice; cost uses the policy's declared unit."""

    arm_id: str
    features: dict[str, float]
    estimated_cost: float = 1.0
    metadata: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.arm_id, str) or not self.arm_id:
            raise ValueError("arm_id must be a nonempty string")
        if not isinstance(self.features, dict) or any(
            not isinstance(key, str) or not key or not _finite(value)
            for key, value in self.features.items()
        ):
            raise ValueError("features must map nonempty names to finite numbers")
        if not _finite(self.estimated_cost) or self.estimated_cost <= 0:
            raise ValueError("estimated_cost must be positive and finite")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def contextual_features(context: dict, choice: dict) -> dict[str, float]:
    """Encode supplied categories and their context-by-choice interactions.

    Pass only information available before the choice. IDs and evaluation
    outcomes are never added automatically. Numeric values remain numeric.
    """
    import json

    def encode(prefix, values):
        result = {}
        for key, value in sorted(values.items()):
            if not isinstance(key, str) or not key:
                raise ValueError("feature keys must be nonempty strings")
            if _finite(value):
                result[f"{prefix}:{json.dumps(key)}"] = float(value)
            elif isinstance(value, (str, bool)) or value is None:
                result[f"{prefix}:{json.dumps([key, value], ensure_ascii=False)}"] = 1.0
            else:
                raise ValueError("context categories must be strings, booleans, null or finite numbers")
        return result

    left, right = encode("context", context), encode("choice", choice)
    interactions = {f"interaction:{json.dumps([a, b])}": x * y
                    for a, x in left.items() for b, y in right.items()}
    if any(not _finite(value) for value in interactions.values()):
        raise ValueError("feature interaction overflow")
    return {**left, **right, **interactions}


class _Policy:
    def __init__(self, *, seed=42, frozen=False, training_partitions=("train", "development", "dev", "search"),
                 training_families=None, reward_range=(-1.0, 1.0), cost_unit="target_calls"):
        if type(seed) is not int or type(frozen) is not bool:
            raise ValueError("seed must be an integer and frozen must be boolean")
        if not isinstance(training_partitions, (list, tuple)) or not training_partitions or any(
            not isinstance(value, str) or not value for value in training_partitions
        ) or set(training_partitions) & {"test", "heldout", "confirmation"}:
            raise ValueError("training_partitions must explicitly name development partitions")
        families = training_families or {}
        if not isinstance(families, dict) or any(
            not isinstance(key, str) or not key or not isinstance(values, (list, tuple))
            or not values or any(not isinstance(value, str) or not value for value in values)
            for key, values in families.items()
        ):
            raise ValueError("training_families must map metadata keys to allowed names")
        if not isinstance(reward_range, (list, tuple)) or len(reward_range) != 2 or any(
            not _finite(value) for value in reward_range
        ) or reward_range[0] >= reward_range[1]:
            raise ValueError("reward_range must contain increasing finite bounds")
        if not isinstance(cost_unit, str) or not cost_unit:
            raise ValueError("cost_unit must be a nonempty name")
        self.seed, self.frozen = seed, frozen
        self.training_partitions = tuple(training_partitions)
        self.training_families = {key: tuple(values) for key, values in families.items()}
        self.reward_range, self.cost_unit = tuple(reward_range), cost_unit
        self.rng = random.Random(seed)
        self.updates = self.selections = self.missing_costs = 0
        self.costs = {}

    @property
    def configuration(self):
        return {"policy": type(self).__name__, "seed": self.seed, "frozen": self.frozen,
                "training_partitions": list(self.training_partitions),
                "training_families": {key: list(value) for key, value in self.training_families.items()},
                "reward_range": list(self.reward_range), "cost_unit": self.cost_unit,
                "cost_model": "per_arm_running_mean_with_declared_estimate_fallback"}

    def _arms(self, arms):
        arms = tuple(arms)
        if not arms or any(not isinstance(arm, Arm) for arm in arms):
            raise ValueError("select requires a nonempty sequence of Arm objects")
        if len({arm.arm_id for arm in arms}) != len(arms):
            raise ValueError("available arm IDs must be unique")
        for arm in arms:
            arm.__post_init__()
        return arms

    def _validate_observation(self, arm, reward, cost):
        if self.frozen:
            raise ValueError("a frozen policy cannot observe evaluation outcomes")
        self._arms([arm])
        if arm.metadata.get("partition") not in self.training_partitions:
            raise ValueError("observation is outside the policy's training partitions")
        if arm.metadata.get("phase") in {"confirmation", "heldout", "test"}:
            raise ValueError("confirmation and held-out outcomes cannot update the policy")
        for key, allowed in self.training_families.items():
            value = arm.metadata.get(key)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"observation is outside the bound training family: {key}")
        if not _finite(reward) or not self.reward_range[0] <= reward <= self.reward_range[1]:
            raise ValueError("reward must lie within the declared finite range")
        if cost is not None and (not _finite(cost) or cost < 0):
            raise ValueError("cost must be nonnegative and finite, or None when unreported")

    def _record(self, arm, cost):
        self.updates += 1
        if cost is None:
            self.missing_costs += 1
        else:
            count, mean = self.costs.get(arm.arm_id, [0, 0.0])
            self.costs[arm.arm_id] = [count + 1, mean + (cost - mean) / (count + 1)]

    def observe(self, arm, *, reward, cost=None):
        self._validate_observation(arm, reward, cost)
        self._record(arm, cost)

    def save(self, path):
        """Persist JSON state atomically; no pickle or executable state."""
        import json
        import os
        import tempfile
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        document = {"schema_version": 1, "configuration": self.configuration,
                    "state": {"rng": self.rng.getstate(), "updates": self.updates,
                              "selections": self.selections, "missing_costs": self.missing_costs,
                              "costs": self.costs, **self._extra_state()}}
        payload = json.dumps(document, allow_nan=False, indent=2) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".policy-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def _extra_state(self):
        return {}

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path
        path = Path(path)
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("policy JSON exceeds the supported size")
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("unsupported policy schema")
        config = dict(data.get("configuration", {}))
        if config.pop("policy", None) != cls.__name__:
            raise ValueError("saved policy type does not match loader")
        if config.pop("cost_model", None) != "per_arm_running_mean_with_declared_estimate_fallback":
            raise ValueError("unsupported cost model")
        for key in ("feature_schema", "feature_normalization", "acquisition"):
            config.pop(key, None)
        policy = cls(**config)
        state = data.get("state")
        if not isinstance(state, dict):
            raise ValueError("missing policy state")
        for name in ("updates", "selections", "missing_costs"):
            value = state.get(name)
            if type(value) is not int or value < 0:
                raise ValueError("invalid policy counter")
            setattr(policy, name, value)
        costs = state.get("costs")
        if not isinstance(costs, dict) or any(
            not isinstance(key, str) or not key or not isinstance(value, list) or len(value) != 2
            or type(value[0]) is not int or value[0] < 1 or not _finite(value[1]) or value[1] < 0
            for key, value in costs.items()
        ):
            raise ValueError("invalid measured cost state")
        if policy.missing_costs + sum(value[0] for value in costs.values()) != policy.updates:
            raise ValueError("cost accounting does not match update count")
        policy.costs = costs
        rng = state.get("rng")
        if not isinstance(rng, list) or len(rng) != 3 or rng[0] != 3 or not isinstance(rng[1], list) or len(rng[1]) != 625 or any(
            type(value) is not int or value < 0 or value > 2**32 - 1 for value in rng[1][:-1]
        ) or type(rng[1][-1]) is not int or not 0 <= rng[1][-1] <= 624 or (rng[2] is not None and not _finite(rng[2])):
            raise ValueError("invalid RNG state")
        policy.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
        policy._restore_extra(state, data["configuration"])
        return policy

    def _restore_extra(self, state, config):
        pass


class UniformPolicy(_Policy):
    """Uniform selection among the currently available choices."""

    def select(self, arms: Sequence[Arm]) -> Arm:
        arms = self._arms(arms)
        self.selections += 1
        return self.rng.choice(arms)


class FixedPolicy(_Policy):
    """Cycle through available arms in a declared order (ID order by default)."""

    def __init__(self, *, order=(), **kwargs):
        super().__init__(**kwargs)
        if not isinstance(order, (list, tuple)) or any(not isinstance(value, str) or not value for value in order) or len(set(order)) != len(order):
            raise ValueError("order must contain distinct nonempty arm IDs")
        self.order = tuple(order)

    @property
    def configuration(self):
        return {**super().configuration, "order": list(self.order)}

    def select(self, arms: Sequence[Arm]) -> Arm:
        arms = self._arms(arms)
        ranks = {name: index for index, name in enumerate(self.order)}
        arms = sorted(arms, key=lambda arm: (ranks.get(arm.arm_id, len(ranks)), arm.arm_id))
        selected = arms[self.selections % len(arms)]
        self.selections += 1
        return selected


class LinearUCB(_Policy):
    """Ridge linear UCB divided by estimated cost; a standard allocation baseline.

    Features use signed SHA-256 hashing plus a bias and L2 normalization. Actual
    costs update per-arm means; unreported costs retain the supplied estimate.
    Negative upper bounds contribute zero useful effect; if all scores are zero,
    select the cheapest available choice. This avoids rewarding higher cost just
    because dividing a negative effect by it moves the score closer to zero.
    NumPy is imported only when a LinearUCB instance is created.
    """

    def __init__(self, *, dimensions=64, regularization=1.0, exploration=1.0,
                 min_cost=1e-9, **kwargs):
        super().__init__(**kwargs)
        if type(dimensions) is not int or not 2 <= dimensions <= 512:
            raise ValueError("dimensions must be an integer between 2 and 512")
        if not _finite(regularization) or regularization <= 0 or not _finite(exploration) or exploration < 0 or not _finite(min_cost) or min_cost <= 0:
            raise ValueError("regularization/min_cost must be positive; exploration must be nonnegative")
        import numpy as np
        self.dimensions, self.regularization = dimensions, regularization
        self.exploration, self.min_cost = exploration, min_cost
        self._np = np
        self.a = np.eye(dimensions) * regularization
        self.b = np.zeros(dimensions)

    @property
    def configuration(self):
        return {**super().configuration, "dimensions": self.dimensions,
                "regularization": self.regularization, "exploration": self.exploration,
                "min_cost": self.min_cost, "feature_schema": "sha256_signed_bias_v1",
                "feature_normalization": "l2_unit",
                "acquisition": "nonnegative_ucb_per_cost_v1"}

    def _vector(self, arm):
        np = self._np
        # Scale before hashing to prevent finite large inputs overflowing sums.
        scale = max(1.0, max((abs(value) for value in arm.features.values()), default=0.0))
        vector = np.zeros(self.dimensions)
        vector[0] = 1.0 / scale
        for name, value in sorted(arm.features.items()):
            digest = hashlib.sha256(name.encode()).digest()
            index = 1 + int.from_bytes(digest[:8], "big") % (self.dimensions - 1)
            vector[index] += (1 if digest[8] & 1 else -1) * (value / scale)
        peak = float(np.max(np.abs(vector)))
        vector /= peak
        return vector / float(np.linalg.norm(vector))

    def select(self, arms: Sequence[Arm]) -> Arm:
        arms = self._arms(arms)
        np = self._np
        vectors = np.stack([self._vector(arm) for arm in arms])
        solved = np.linalg.solve(self.a, vectors.T).T
        means = solved @ self.b
        uncertainty = np.sqrt(np.maximum(0, np.sum(vectors * solved, axis=1)))
        costs = np.array([max(self.min_cost, self.costs.get(arm.arm_id, [0, arm.estimated_cost])[1]) for arm in arms])
        optimism = np.maximum(0.0, means + self.exploration * uncertainty)
        scores = optimism / costs
        if not np.all(np.isfinite(scores)):
            raise ValueError("allocation score overflow")
        best_mask = np.isclose(scores, scores.max(), rtol=1e-12, atol=1e-12)
        if float(scores.max()) == 0.0:
            best_mask &= costs == costs.min()
        best = np.flatnonzero(best_mask).tolist()
        self.selections += 1
        return arms[self.rng.choice(best)]

    def observe(self, arm, *, reward, cost=None):
        self._validate_observation(arm, reward, cost)
        vector = self._vector(arm)
        next_a = self.a + self._np.outer(vector, vector)
        next_b = self.b + reward * vector
        if not self._np.all(self._np.isfinite(next_a)) or not self._np.all(self._np.isfinite(next_b)):
            raise ValueError("policy update overflow")
        self.a, self.b = next_a, next_b
        self._record(arm, cost)

    def _extra_state(self):
        return {"a": self.a.tolist(), "b": self.b.tolist()}

    def _restore_extra(self, state, config):
        if (config.get("feature_schema") != "sha256_signed_bias_v1"
                or config.get("feature_normalization") != "l2_unit"
                or config.get("acquisition") != "nonnegative_ucb_per_cost_v1"):
            raise ValueError("unsupported saved feature schema")
        np = self._np
        a, b = np.asarray(state.get("a")), np.asarray(state.get("b"))
        if a.shape != (self.dimensions, self.dimensions) or b.shape != (self.dimensions,) or a.dtype.kind not in "ifu" or b.dtype.kind not in "ifu":
            raise ValueError("saved ridge arrays have invalid dimensions or types")
        a, b = a.astype(float), b.astype(float)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.allclose(a, a.T, rtol=1e-12, atol=1e-12):
            raise ValueError("saved ridge arrays must be finite and symmetric")
        if np.linalg.eigvalsh(a).min() < self.regularization * (1 - 1e-8):
            raise ValueError("saved ridge matrix violates its regularization bound")
        if not math.isclose(float(np.trace(a)), self.dimensions * self.regularization + self.updates, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("saved ridge matrix does not match update count")
        self.a, self.b = a, b
