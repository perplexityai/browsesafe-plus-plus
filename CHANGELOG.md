# Changelog

## 0.2.0

Leaderboard submissions for the test split, from the same gated dataset as the training split.

- The label-free test inputs (one page of each test pair, the attacked page or its control) are the `test` split of `perplexity-ai/browsesafe-plus-plus`, in one configuration with `train` (`test/data/test-*.parquet`, `test/screenshots/<entry_id[:2]>/<entry_id>.jpg`, `test/screenshots/manifest.jsonl`). Test rows use the training columns with an opaque `entry_id`, `split="test"` and every label and annotation null. `TEST_DATASET` equals `DEFAULT_DATASET`, and `BROWSESAFE_DATASET` applies to both splits.
- `load_test_inputs()` loads the test pages (`TestPage`: opaque `id` taken from `entry_id`; content, view, benign task) from the Hub, downloading only the files under `test/`, or from a local copy (the dataset directory or its `test/` directory). `bs.load()` never reads them.
- The dataset is gated (one access request covers both splits). When the Hub refuses access, loading raises `PermissionError` with the dataset page to request access on and the login step (`hf auth login` or `HF_TOKEN`).
- Licences: the training split is CC BY 4.0 and the test inputs are under an evaluation-only licence (`LICENSE.md` in the dataset repository).
- `predict_test()` / `predict_test_async()` run a detector on them and return records in the leaderboard's format; `write_predictions()` validates the records and writes the JSON Lines file to upload.
- `browsesafe predict --predict module:function --view VIEW ... --output FILE` does both for one or more views.
- `bs.load(split="test")` against the Hub still raises an error, which now points to `load_test_inputs` and the leaderboard Space.
- The dataset shards embed each page's JPEG in the `screenshot` column; the screenshot view of `load`, `iter_batches` and `load_test_inputs` reads the embedded images. `Page.metadata["screenshot"]` holds only the path.
- `browsesafe_pp.testing.write_test_fixture()` (`python -m browsesafe_pp.testing DIR --test-inputs`) writes an offline fixture in the test-input layout (`test/data` in the training schema, `test/screenshots`, plus a `labels.parquet` for local checks); it can share a directory with a `write_fixture()` copy, whose shards now carry the published 26 columns in order with their Hugging Face features. Both embed a tiny JPEG per page.
- New guide: [Submitting to the leaderboard](docs/leaderboard.md).

## 0.1.0

First public version of the SDK.

- Loads the public training split by default (`split="train"`); the test split is reserved for the leaderboard, and requesting it from the Hub raises an error that says so.
- Paired loading from the Hugging Face Hub (`BROWSESAFE_DATASET`, `revision=`) or a local directory of `members-*.parquet` shards (`source=`), with seeded sampling, metadata filters, exclusion accounting and bounded-memory batching (`iter_batches`).
- Views: HTML, Markdown, accessibility tree and full-page screenshots.
- `describe()` and `browsesafe info` for dataset summaries.
- Detection scoring in `browsesafe_pp.evaluation.classification` (`evaluate`, `evaluate_async`, `score`, `DetectionReport`); `browsesafe_pp.evaluate` is the top-level shortcut.
- `ModelDetector` with the BrowseSafe 1.0 classification prompt and tool contract.
- One OpenAI-compatible provider (`OpenAICompatible`) configured by `BROWSESAFE_MODEL_ENDPOINT`, `BROWSESAFE_API_KEY` and `BROWSESAFE_MODEL`, shared by evaluation and the browser target.
- Adaptive attack framework: contexts, seeds from real injections, `PortfolioSearch`, bandit policies (`UniformPolicy`, `FixedPolicy`, `LinearUCB`), budgets, judges, confirmations, event logs and `reports.aggregate`.
- Optional Playwright browser target (`[browser]` extra).
- `browsesafe evaluate` with a custom `module:function` detector or `--model`.
- Synthetic offline fixture (`python -m browsesafe_pp.testing`).
