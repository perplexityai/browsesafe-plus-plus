# BrowseSafe++ SDK documentation

BrowseSafe++ measures whether a detector or an agent can tell a web page that carries an indirect prompt injection from the same page without it. The SDK covers three jobs:

1. **Load the data**: complete attack/control pairs in the view you choose. → [Data](data.md)
2. **Evaluate detectors**: score any classifier, or an OpenAI-compatible model with the BrowseSafe 1.0 prompt. → [Classification](classification.md)
3. **Run adaptive attacks**: budgeted campaigns against your own agent, seeded with real injections. → [Adaptive attacks](adaptive.md)
4. **Get on the leaderboard**: serve your detector or agent as an endpoint; SII evaluates it on the private test split. → [Leaderboard](leaderboard.md)

| Page | Contents |
|---|---|
| [Installation](installation.md) | Requirements, extras, offline fixture |
| [Data](data.md) | Schema, fields, splits, views, screenshots, filters, batching |
| [Classification](classification.md) | Prompt, detector contract, metrics, CLI and Python |
| [Adaptive attacks](adaptive.md) | Contexts, seeds, policies, budgets, judges, reports; worked example |
| [Leaderboard](leaderboard.md) | Public `train` vs private `test`, the `/predict` endpoint contract, submitting verified results |
| [Providers](providers.md) | OpenAI-compatible endpoints and environment variables |
| [API reference](api.md) | Public classes and functions |
| [Contributing](contributing.md) | Development setup and conventions |
| [Changelog](../CHANGELOG.md) | Release history |

## Concepts in one minute

- A **pair** is two pages sharing a five-digit index: `00042_malicious` (label `yes`) and `00042_benign` (label `no`). The benign page is the control: the same page with the injection removed.
- A **view** is the representation a model sees: `html`, `markdown`, `accessibility` or `screenshot`.
- A **benchmark** is a seeded sample of pairs plus a **manifest** that records the source, revision, filters, selected pair ids and every exclusion.
- A **detector** maps a `DetectorInput` (content and view only; no labels) to a `Prediction`. The same two dataclasses are the JSON wire format of `browsesafe serve` and `EndpointDetector`.
- The public dataset ships the **`train`** partition; the **`test`** partition is private and evaluated by SII for the [leaderboard](leaderboard.md).
- A **campaign** attacks one **context** (page, task, goal, insertion slot) within a **budget**, judged by your **judge**.
