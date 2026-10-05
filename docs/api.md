# API reference

Everything below is importable from `browsesafe_pp` (as `bs`) or `browsesafe_pp.adaptive` (as `a`). Names not listed here are internal.

```python
import browsesafe_pp as bs
from browsesafe_pp import adaptive as a
```

## Data

| Name | Signature | Description |
|---|---|---|
| `bs.load` | `load(*, task="detection", split="train", view="html", source=None, dataset=None, revision="main", token=None, n_pairs=None, seed=42, filters=None, offline=False, split_column="split") -> Benchmark` | Load complete attack/control pairs. `source` is a local shard directory; otherwise the Hub dataset is used. `split=None` loads all splits; `split="test"` raises a `ValueError` unless the source contains the private split. |
| `bs.iter_batches` | `iter_batches(*, batch_pairs=64, order="selection", max_spool_bytes=10 GiB, spool_dir=None, **load_args) -> Iterator[Benchmark]` | The same pairs as `load`, in bounded batches. |
| `bs.describe` | `describe(*, source=None, dataset=None, revision="main", token=None, offline=False, split_column="split") -> dict` | Counts of pages, pairs, splits and page types (metadata columns only). |
| `bs.dataset_id` | `dataset_id(dataset=None) -> str` | The argument, else `$BROWSESAFE_DATASET`, else `bs.DEFAULT_DATASET`. |
| `bs.VIEWS`, `bs.ALL_VIEWS`, `bs.SCREENSHOT_VIEW` | constants | Text views (`html`, `markdown`, `accessibility`), all views, and `"screenshot"`. |
| `bs.DEFAULT_DATASET`, `bs.DEFAULT_REVISION` | constants | Default Hub id and revision. |
| `bs.PRIVATE_SPLITS` | constant | `frozenset({"test"})`: partitions not in the public dataset, evaluated by SII ([Leaderboard](leaderboard.md)). |

### Classes

| Class | Fields and methods |
|---|---|
| `Benchmark` | `pairs: tuple[Pair, ...]`, `manifest: dict`; `to_records()` yields one supervised record (content plus label) per page. |
| `Pair` | `pair_id`, `malicious: Page`, `benign: Page`; `metadata` (pair-level annotations live on the malicious member). |
| `Page` | `entry_id`, `content`, `view`, `task`, `injected: bool`, `metadata: dict`, `evidence: tuple[str, ...]` (injection quotes; empty for controls). For the screenshot view, `content` is the image path. |
| `DetectorInput` | `content`, `view`, `task`: what a detector sees. Labels and annotations are excluded. Also the JSON request body of `POST /predict`. |
| `Prediction` | `injected: bool | None` (`None` abstains), `score: float | None` in [0, 1]. Also the JSON response body of `POST /predict`. |

## Classification

| Name | Signature | Description |
|---|---|---|
| `bs.evaluate` | `evaluate(benchmark, predict, *, evaluator="self", run_id=None) -> DetectionReport` | Call a synchronous `predict(DetectorInput)` once per page (no retries). It may return a `Prediction`, a `bool` or `None`. |
| `bs.evaluate_async` | `evaluate_async(benchmark, predict, *, concurrency=8, evaluator="self", run_id=None) -> DetectionReport` | The same for an async (or sync) predictor, with bounded concurrency. |
| `bs.ModelDetector` | `ModelDetector(provider, prompt="browsesafe-1.0", *, max_tokens=4096, temperature=0.0, use_tools=True)` | Async detector that sends the classification prompt to an OpenAI-compatible model; `messages(request)` shows the exact request. |
| `bs.EndpointDetector` | `EndpointDetector(url, *, token=None, timeout=60, retries=2, backoff=0.5, headers=None, transport=None)` | Synchronous detector that POSTs the `DetectorInput` to a `/predict` endpoint (the `browsesafe serve` contract) and returns its `Prediction`; retries timeouts and 5xx; `configuration` never includes the token; `close()`. Needs `httpx`. |
| `bs.evaluation.get_classification_prompt` | `get_classification_prompt(name="browsesafe-1.0") -> ClassificationPrompt` | The built-in prompt (alias `"default"`). |
| `bs.evaluation.ClassificationPrompt` | `ClassificationPrompt(name, text, output_format="label", source=None, note=None)` | A custom prompt for `ModelDetector`. |
| `bs.evaluation.score` | `score(benchmark, outcomes, execution=None, *, evaluator="self", run_id=None) -> DetectionReport` | Score precomputed per-page `(status, injected, score)` outcomes in benchmark order. |
| `DetectionReport` | `manifest`, `metrics`, `predictions`; `save(path)` | `save` writes `manifest.json`, `metrics.json` and `predictions.jsonl` (no page text) to a new directory. The manifest includes `partition` (split or `"all"`), `evaluator` and `run_id`. |

## Providers

| Name | Description |
|---|---|
| `bs.OpenAICompatible(endpoint=None, model=None, api_key=None, *, timeout=120, token_limit_field="max_completion_tokens", prices_per_million=None, extra_body=None, transport=None)` | Async Chat Completions client. Arguments default to `BROWSESAFE_MODEL_ENDPOINT`, `BROWSESAFE_MODEL` and `BROWSESAFE_API_KEY`. `await chat(messages, *, max_tokens=1024, temperature=None, tools=None, tool_choice=None, response_format=None, seed=None) -> ChatResponse`; `await aclose()`; `configuration` never includes the key. |
| `bs.ProviderError` | Raised on transport or API errors. |
| `bs.providers.ChatResponse`, `ToolCall`, `Usage` | Response text, tool calls and token usage. |
| `bs.providers.decode_json(text) -> dict` | Parse a JSON object from model output, tolerating code fences. |

## Adaptive attacks

| Name | Signature | Description |
|---|---|---|
| `a.run` | `run(context, *, attacker, target_factory, judge, output_dir, budget=Budget(), seeds=(), mode="replay", feedback_mode="outcome", confirmations=1, require_clean_success=True, stop_on_success=True, protocol_id="caller-defined-v1") -> CampaignResult` | One campaign against one context. See [Adaptive attacks](adaptive.md). |
| `a.from_pair` | `from_pair(benchmark, pair, *, slot, goal=None) -> Context` | A context from a pair's benign page, task and goal. |
| `a.local_callback` | decorator | Declares a callback that never calls a model (its usage is known to be zero). |
| `a.seeds.from_benchmark` | `from_benchmark(benchmark, *, n=8, match=None, query="", split="train", seed=42) -> tuple[Seed, ...]` | Real injection fragments as seeds. |
| `a.reports.aggregate` | `aggregate(campaigns) -> dict` | Combine results, summaries or directories. |
| `a.reports.load_summary` | `load_summary(directory) -> dict` | Read `summary.json`. |
| `a.reports.wilson` | `wilson(successes, total, z=1.96) -> list[float] | None` | Wilson score interval. |
| `a.contextual_features` | `contextual_features(context: dict, choice: dict) -> dict[str, float]` | Categorical features and their interactions for allocation policies. |

### Contracts

| Class | Fields |
|---|---|
| `Context` | `context_id`, `task`, `observation`, `goal`, `slot`, `view="markdown"`, `source_revision`, `source_entry_id`, `source_family`, `split="development"`, `attacker_context`, `surfaces=()`, `checkpoint=None` |
| `Slot` | `start`, `end`, `name="page_insertion"` (half-open offsets) |
| `Surface` | `surface_id`, `slot`, `description=""` |
| `BrowserCheckpoint` | `checkpoint_id`, `tool_call_id`, `tool_name`, `url`, `viewport`, `visible_history` |
| `Seed` | `seed_id`, `payload`, `source_revision`, `source_entry_id`, `attack_identified_ts` |
| `Candidate` | `payload`, `strategy="caller_proposed"`, `parent_attempt_id`, `source_seed_id`, `surface_id`, `derived_from_seed_id`, `usage` |
| `TargetInput` | `task`, `observation`, `view`, `trust="untrusted_observation"`, `checkpoint`, `surface_id`, `trial_seed`, `max_model_calls`, `max_tokens`, `max_cost_usd` |
| `TargetResult` | `output`, `events=()`, `detected`, `injection_observed`, `detector_score`, `usage` |
| `Target` | protocol: `async run(TargetInput) -> TargetResult`, `async aclose()` |
| `Trial` | `context`, `candidate` (`None` for the baseline), `result`, `phase` |
| `Verdict` | `task_success`, `attack_success`, `attack_valid=None`, `evidence=""`, `progress=None`, `usage=None` |
| `Feedback` | `attempt_id`, `candidate`, `status`, `task_success`, `attack_success`, `attack_valid`, `detected`, `injection_observed`, `target_output`, `confirmation_passed`, `progress`, `events`, ... (fields hidden by `feedback_mode` are `None`) |
| `Budget` | `max_candidates=8`, `max_attacker_calls=8`, `max_target_calls=10`, `max_judge_calls=10`, `max_seconds=600`, `call_timeout=60`, `cleanup_timeout=5`, `max_payload_chars=20000`, `max_model_calls`, `max_tokens`, `max_cost_usd` |
| `RemainingBudget` | `attacker_calls`, `target_calls`, `judge_calls`, `seconds` |
| `Usage` | `model_calls=0`, `input_tokens`, `output_tokens`, `cost_usd`, `cost_basis`, ... |
| `CampaignResult` | `output_dir`, `summary`, `history` |

### Policies

| Class | Description |
|---|---|
| `PortfolioSearch(mutate, *, seed=42, parent_count=8, restart_every=8, initial_candidates=())` | Quality-diversity attacker; calls `async mutate(SearchRequest) -> Candidate` once per proposal. |
| `SearchRequest` | `context`, `parents`, `recent`, `remaining`, `restart`, `seeds` |
| `Arm(arm_id, features, estimated_cost=1.0, metadata={})` | One allocation choice. |
| `UniformPolicy(*, seed=42, frozen=False, training_partitions=(...), training_families=None, reward_range=(-1, 1), cost_unit="target_calls")` | Random baseline. |
| `FixedPolicy(*, order=(), **kwargs)` | Declared order. |
| `LinearUCB(*, dimensions=64, regularization=1.0, exploration=1.0, min_cost=1e-9, **kwargs)` | Cost-aware contextual bandit (needs NumPy). |

All policies provide `select(arms) -> Arm`, `observe(arm, *, reward, cost=None)`, `save(path)`, `load(path)` (class method) and `configuration`.

## Command line

```text
browsesafe info [--source DIR] [--dataset ID] [--revision REV] [--offline]
browsesafe evaluate [--source DIR] [--dataset ID] [--revision REV] [--offline]
                    [--split SPLIT] [--view {html,markdown,accessibility,screenshot}]
                    [--n-pairs N] [--seed S] [--output DIR]
                    (--predict MODULE:FUNCTION | --model MODEL)
                    [--endpoint URL] [--concurrency N] [--max-tokens N]
                    [--evaluator NAME] [--run-id ID]
browsesafe serve --predict MODULE:FUNCTION [--host HOST] [--port PORT]
```

`browsesafe serve` exposes `POST /predict` (`{"content", "view", "task"}` → `{"injected", "score"}`) and `GET /health` using only the standard library; `browsesafe_pp.cli.make_server(predict, host, port)` returns the underlying `ThreadingHTTPServer` for embedding or tests.

`python -m browsesafe_pp.testing DIR` writes a small offline fixture in the dataset's format.
