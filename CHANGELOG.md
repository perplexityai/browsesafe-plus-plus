# Changelog

## 0.2.0

The leaderboard `test` split is private; the public dataset ships `train` only.

- `load()`, `iter_batches()` and `browsesafe evaluate` default to `split="train"`. New `PRIVATE_SPLITS = frozenset({"test"})`; requesting a private split that a source does not contain raises a `ValueError` that points to `docs/leaderboard.md`. Local sources and the internal dataset still load `test`.
- Endpoint evaluation contract: `browsesafe serve --predict module:function` exposes a detector as `POST /predict` / `GET /health` (standard library only), and `EndpointDetector` (needs `httpx`) calls such an endpoint with a bearer token, timeout and bounded retries on timeouts and 5xx.
- Report provenance: the `DetectionReport` manifest records `partition`, `evaluator` (default `"self"`) and `run_id`; `evaluate`, `evaluate_async` and `score` take `evaluator=` / `run_id=`; the CLI takes `--evaluator` / `--run-id`.
- Docs: new `docs/leaderboard.md` (two tracks, the `/predict` and OpenAI-compatible agent contracts, why the test set is not distributed); the adaptive budget caps are documented as the per-run cap for leaderboard campaigns; project URLs moved to <https://browsesafe.perplexity.ai/>.

## 0.1.0

First public version of the SDK.

- Paired loading from the Hugging Face Hub (`BROWSESAFE_DATASET`, `revision=`) or a local directory of `members-*.parquet` shards (`source=`), with seeded sampling, metadata filters, exclusion accounting and bounded-memory batching (`iter_batches`).
- Views: HTML, Markdown, accessibility tree and full-page screenshots.
- `describe()` and `browsesafe info` for dataset summaries.
- Detection scoring in `browsesafe_pp.evaluation.classification` (`evaluate`, `evaluate_async`, `score`, `DetectionReport`); `browsesafe_pp.dataset.evaluate` remains importable.
- `ModelDetector` with the BrowseSafe 1.0 classification prompt and tool contract.
- One OpenAI-compatible provider (`OpenAICompatible`) configured by `BROWSESAFE_MODEL_ENDPOINT`, `BROWSESAFE_API_KEY` and `BROWSESAFE_MODEL`, shared by evaluation and the browser target.
- Adaptive attack framework: contexts, seeds from real injections, `PortfolioSearch`, bandit policies (`UniformPolicy`, `FixedPolicy`, `LinearUCB`), budgets, judges, confirmations, event logs and `reports.aggregate`.
- Optional Playwright browser target (`[browser]` extra).
- `browsesafe evaluate` with a custom `module:function` detector or `--model`.
- Synthetic offline fixture (`python -m browsesafe_pp.testing`).
