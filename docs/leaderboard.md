# Leaderboard

The [BrowseSafe++ leaderboard](https://browsesafe.perplexity.ai/leaderboard/) reports detection and adaptive-attack results on the **private test split**. This page explains what is public, what is private and why, and how to get a verified row.

## Partitions

| Partition | Pairs | Where it lives | Who can run it |
|---|---|---|---|
| `train` | 29,500 (59,000 pages) | [perplexity-ai/browsesafe-plus-plus](https://huggingface.co/datasets/perplexity-ai/browsesafe-plus-plus), public, with screenshots | anyone, locally |
| `test` | 9,175 (18,350 pages) | `perplexity-ai/browsesafe-plus-plus-internal`, private | SII, against your endpoint |

The public dataset contains **only** the `train` partition. The SDK defaults to `split="train"`, and requesting `split="test"` against a source that does not contain it raises a `ValueError` pointing here (`bs.PRIVATE_SPLITS`). Local sources that do contain test rows (for example the offline fixture, or the internal dataset selected with `BROWSESAFE_DATASET`) load normally.

### Why the test split is not distributed

BrowseSafe++ is a labelled binary benchmark whose two arms differ only by the injection. Both properties make a public test set easy to solve without building a detector:

- **Labels are cheap to produce by hand.** Every page carries one of two labels, so a released test set can be annotated, memorised or used to tune thresholds in a few hours, after which test scores stop measuring generalisation.
- **Attack/control pairs reveal the injection by diffing.** The control page is the attacked page with the injection removed. With both members in hand, `diff` recovers the injected text exactly, which turns the detection task into a lookup.

Keeping the test pages, their screenshots and their labels inside SII preserves the benchmark as a held-out measurement. Models that trained on `train` (including BrowseSafe++ Guard) are evaluated on `test` the same way as everyone else.

## Two tracks

### Local results on `train` (self-reported)

Anyone can run the public split:

```bash
browsesafe evaluate --predict my_detector:predict --split train --n-pairs 0 --output runs/train
```

These numbers are useful for development and for comparing with the paper's `train` rows, but they are **not shown on the main leaderboard**: the detector may have seen the pages, and nobody can verify the run. Share them in your own paper or repository together with the report directory (`manifest.json` records the dataset revision, `selection_digest`, `partition: "train"` and `evaluator: "self"`).

### Verified leaderboard results on `test`

SII runs your system against the private split and publishes the row. You expose an endpoint; the private pages never leave SII.

**Detection track.** Serve your detector with the `/predict` contract:

```bash
pip install "browsesafe-pp[eval]"
browsesafe serve --predict my_detector:predict --host 0.0.0.0 --port 8000
```

`browsesafe serve` wraps the same `predict(DetectorInput) -> Prediction | bool | None` function that `browsesafe evaluate` calls locally, so what you tested is what gets scored. Any server that implements the contract works:

| Route | Request | Response |
|---|---|---|
| `POST /predict` | `{"content": str, "view": str, "task": str \| null}` | `{"injected": bool \| null, "score": float \| null}` (`injected: null` abstains; `score` in [0, 1]) |
| `GET /health` | | `{"status": "ok", "sdk_version": "..."}` |

A `500` response is scored as an `error` and a `422` response as an `invalid_prediction`; neither becomes a label. Requests carry `Authorization: Bearer <token>` when you give SII a token. SII calls the endpoint with `bs.EndpointDetector`, which retries timeouts and 5xx responses a bounded number of times and otherwise applies the usual one-call-per-page protocol. You can check your deployment the same way:

```python
import browsesafe_pp as bs

detector = bs.EndpointDetector("https://detector.example.com/predict", token="...")
report = bs.evaluate(bs.load(split="train", n_pairs=50), detector)
```

**Agent track (adaptive attacks).** Expose your agent as an **OpenAI-compatible chat-completions endpoint with tool calling**, the interface `bs.OpenAICompatible` already speaks. SII runs the adaptive campaign against it under a fixed `a.Budget` (candidates, target calls, seconds, and optionally model calls, tokens and cost); see [Adaptive attacks](adaptive.md#leaderboard-evaluation).

**Submitting.** Send the endpoint URL, credentials, the view(s) you support and a short system description through the [submission form](https://browsesafe.perplexity.ai/leaderboard/#submit). SII runs the private split, writes the report with `evaluator: "sii"`, a `run_id` and the dataset revision, and publishes the row with those identifiers so every number can be traced to one verified run. Credentials are used only for that run and discarded afterwards.

## What a row records

| Field | Source |
|---|---|
| `partition` | `"test"` for leaderboard rows; `"train"` or `"all"` for local reports |
| `evaluator` | `"sii"` for verified rows; `"self"` for local reports |
| `run_id` | assigned by SII at evaluation time |
| `revision`, `selection_digest` | the dataset commit and the exact pair selection |
| metrics | as defined in [Classification](classification.md#metrics) and [Adaptive attacks](adaptive.md#summary-fields) |

All of these are in `manifest.json` and `metrics.json` of the report `DetectionReport.save()` writes, for local and verified runs alike.
