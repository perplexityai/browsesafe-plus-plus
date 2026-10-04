# Changelog

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
