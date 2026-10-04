# Adaptive attacks

Static benchmarks measure a detector or agent against fixed injections. An adaptive evaluation asks a harder question: *given feedback and a budget, can an attacker find an injection that makes this agent deviate on this page?* The `browsesafe_pp.adaptive` module provides the campaign loop and its bookkeeping. You supply the agent under test (the **target**), the outcome **judge** and the **attacker** (proposal strategy); the SDK enforces budgets, isolation, confirmation and logging.

```python
from browsesafe_pp import adaptive as a
```

## Concepts

### Context

A `Context` is one attack opportunity: the page the agent will read, the user's task, the attacker's goal and where payloads go.

```python
context = a.Context(
    context_id="00042", task="Summarize the reviews", observation=page_text,
    goal="Make the summary recommend product X", slot=a.Slot(start, end),
    view="markdown", split="train",
)
```

- `slot` is a half-open character range in `observation` that a payload replaces (`start == end` inserts). Additional named insertion points can be declared with `surfaces=(a.Surface("footer", a.Slot(i, i)),)`; a candidate picks one with `surface_id`.
- `a.from_pair(benchmark, pair, slot=...)` builds a context from a loaded pair: the **benign** page, its task and the pair's malicious goal. The injected page is never shown to the target.
- `attacker_context` carries anything the attacker may know (for example the page type). It never reaches the target.

### Target

The target is the system under test. `target_factory` returns a fresh object per trial, so no state leaks between attempts:

```python
class MyAgent:
    async def run(self, request: a.TargetInput) -> a.TargetResult:
        answer = await my_agent(task=request.task, page=request.observation)
        return a.TargetResult(output=answer, detected=None)

    async def aclose(self) -> None:
        ...
```

`TargetInput` contains only what a deployed agent would see: task, observation, view and trust level. It excludes the attacker goal, labels, seed identity and evaluator annotations. `TargetResult.detected` reports a built-in guard decision, if the agent has one; `events` can carry tool calls for a trace-level judge.

### Judge

The judge decides each trial. It sees the full `Trial` (context, candidate, target result, phase) and returns a `Verdict`:

```python
async def judge(trial: a.Trial, target) -> a.Verdict:
    if trial.candidate is None:                 # clean baseline
        return a.Verdict(task_success=solved(trial.result.output), attack_success=None)
    hit = goal_reached(trial.result.output, trial.context.goal)
    return a.Verdict(task_success=solved(trial.result.output), attack_success=hit,
                     attack_valid=True, evidence="why" if hit else "")
```

`attack_valid=False` marks a payload that does not count (for example, it gives away the answer). `progress` (0 to 1) lets a search rank near misses.

### Attacker

An attacker is an async callable `attacker(context, history, remaining) -> Candidate`. `history` is the feedback from earlier trials and `remaining` the unspent budget. What feedback contains depends on `feedback_mode`:

| `feedback_mode` | The attacker sees |
|---|---|
| `"outcome"` | verdicts only (success, validity, detection) |
| `"output"` | plus the target's answer |
| `"trace"` | plus target events and judge evidence |

Any strategy fits this signature: a fixed list, an LLM red-teamer, or the built-in `PortfolioSearch`.

### Seeds

Seeds are injections taken unchanged from the dataset, with their source entry, revision, and detection time:

```python
pool = bs.load(split="train", task="evidence", n_pairs=500)
seeds = a.seeds.from_benchmark(pool, n=8, query=f"{context.task} {context.goal}")
```

Take seeds and contexts from disjoint pairs. The paper draws contexts from the test split, which is reserved for the leaderboard; with the public data, hold out some training pairs as contexts and draw seeds from the rest, as [adaptive_seeded.py](../examples/adaptive_seeded.py) does. In `mode="replay"` every candidate must be an unchanged registered seed (a non-adaptive baseline); in `mode="generate"` candidates may also be new payloads, which are recorded as `adaptive_generated`.

### Budget

```python
budget = a.Budget(max_candidates=8, max_attacker_calls=8, max_target_calls=10,
                  max_judge_calls=10, max_seconds=600, call_timeout=60,
                  max_model_calls=None, max_tokens=None, max_cost_usd=None)
```

Every call is charged, including failures and timeouts. Callbacks report model usage through `usage=a.Usage(...)` on candidates, results and verdicts. A callback that never calls a model is marked with `@a.local_callback`; an unmarked callback that reports no usage is counted as unreported, so the summary shows a lower bound rather than a silent zero.

## Running a campaign

```python
result = await a.run(
    context,
    attacker=attacker, target_factory=MyAgent, judge=judge,
    seeds=seeds, mode="generate", feedback_mode="output",
    confirmations=2, budget=budget, output_dir="runs/00042",
)
print(result.summary["confirmed_success"], result.summary["stop_reason"])
```

The loop:

1. **Baseline.** The target runs on the clean page. If it fails the task, the context is not eligible (`require_clean_success=True`, the default) and no attack is attempted.
2. **Propose.** The attacker returns a candidate; the payload is inserted into the slot (or the chosen surface).
3. **Trial.** A fresh target runs on the modified page; the judge scores it; feedback is appended to the history.
4. **Confirm.** A judged success is re-run `confirmations` times on fresh targets. Only a success that repeats counts as confirmed.
5. **Stop** on confirmed success (`stop_on_success=True`) or when any budget is exhausted.

`output_dir` must not exist. The run writes `events.jsonl` (every call, in order), `manifest.json` (context, seeds, budget, configuration) and `summary.json`. The benchmark data is never modified (`frozen_benchmark_modified: false`).

### Summary fields

| Field | Meaning |
|---|---|
| `baseline_task_success` | Did the target solve the clean page? |
| `candidate_trials`, `generated_trials`, `seed_replay_trials` | Trials run, split by origin |
| `attack_successes`, `attack_success_rate_judged` | Judged successes and their rate over judged trials |
| `attack_success_rate_attempted_lower_bound` | Successes over all attempted trials (unknowns count as failures) |
| `attacked_task_success_rate_judged` | How often the user task still succeeded under attack |
| `confirmed_success`, `first_success_attempt_id` | Whether a success survived confirmation, and which attempt |
| `calls`, `usage` | Calls charged per role; model calls, tokens and cost (with lower bounds) |
| `stop_reason` | Why the campaign ended, e.g. `attack_success_confirmed`, `clean_baseline_not_successful`, `candidate_budget_exhausted` or `call_or_time_budget_exhausted` |

## Search policies

### `PortfolioSearch`

A quality-diversity search around your mutator:

```python
async def mutate(request: a.SearchRequest) -> a.Candidate:
    parent = request.parents[0].candidate if request.parents else request.seeds[0]
    return a.Candidate(payload=rewrite(parent.payload, request.recent), strategy="llm_rewrite",
                       parent_attempt_id=request.parents[0].attempt_id if request.parents else None)

attacker = a.PortfolioSearch(mutate, initial_candidates=tuple(
    a.Candidate(payload=s.payload, source_seed_id=s.seed_id, strategy="seed_replay") for s in seeds))
```

The archive keeps the best trial per cell (surface, strategy, payload-length bucket). Each call picks one elite plus rank-weighted parents and calls `mutate` exactly once. Initial candidates are tried first; every `restart_every` proposals the mutator gets no parents, which forces fresh starts.

### Allocation across contexts: `UniformPolicy`, `FixedPolicy`, `LinearUCB`

When a study has more contexts, seeds or strategies than budget, an allocation policy chooses the next **arm**:

```python
policy = a.LinearUCB(exploration=1.0, seed=42)
arms = [a.Arm(arm_id=s.seed_id, features=a.contextual_features(
            {"page_type": ctx_page_type}, {"payload_length": len(s.payload)}), estimated_cost=1.0,
            metadata={"partition": "train"})          # policies learn only from training partitions
        for s in seeds]
arm = policy.select(arms)
...                                    # run the trial for that arm
policy.observe(arm, reward=1.0 if success else 0.0, cost=target_calls)
policy.save("runs/policy.json")
```

- `UniformPolicy` is the random baseline; `FixedPolicy(order=...)` cycles in a declared order.
- `LinearUCB` is a ridge contextual bandit whose upper bound is divided by expected cost.
- `contextual_features` encodes only information available before the choice; IDs and outcomes are never added automatically.
- Policies learn only from training partitions (`training_partitions`); `frozen=True` stops learning for held-out evaluation. `save` / `load` make a study resumable.

## Reports

```python
report = a.reports.aggregate(["runs/00042", "runs/00043", result])
```

`aggregate` combines campaign results, summaries or directories. It reports eligible contexts, contexts whose baseline failed, attack success with known/unknown counts and a 95% Wilson interval, and total usage. `a.reports.wilson(k, n)` gives the interval on its own.

## Worked examples

- [`examples/adaptive_local.py`](../examples/adaptive_local.py): one campaign against a deliberately brittle local target; no network, no model.
- [`examples/adaptive_seeded.py`](../examples/adaptive_seeded.py): held-out training contexts, seeds from other training pairs and `PortfolioSearch`; runs on the offline fixture.

```bash
python -m browsesafe_pp.testing ./fixture
python examples/adaptive_seeded.py --source ./fixture --output runs/adaptive-seeded
```

The local targets are stand-ins that show the loop; their numbers are not benchmark measurements. For a real study, replace the target with your agent, the judge with an outcome check for the goal, and the mutator with your attack model.

## Good practice

- Keep contexts and seeds on disjoint pairs, and let policies learn only from training partitions.
- Keep the judge independent of the attacker (a different model, or a deterministic check).
- Report confirmed successes with their Wilson interval and the full budget, including baseline-failed contexts.
- Keep `events.jsonl`: it is the evidence for every number in `summary.json`.
