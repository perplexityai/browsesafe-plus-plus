# BrowseSafe++ SDK documentation

BrowseSafe++ measures whether a detector or an agent can tell a web page that carries an indirect prompt injection from the same page without it. The SDK covers three jobs:

1. **Load the data**: complete attack/control pairs in the view you choose. → [Data](data.md)
2. **Evaluate detectors**: score any classifier, or an OpenAI-compatible model with the BrowseSafe 1.0 prompt. → [Classification](classification.md)
3. **Run adaptive attacks**: budgeted campaigns against your own agent, seeded with real injections. → [Adaptive attacks](adaptive.md)

To score a detector on the test split, run it on the label-free test pages and upload the predictions to the leaderboard Space. → [Submitting to the leaderboard](leaderboard.md)

| Page | Contents |
|---|---|
| [Installation](installation.md) | Requirements, extras, offline fixture |
| [Data](data.md) | Schema, fields, splits, views, screenshots, filters, batching |
| [Classification](classification.md) | Prompt, detector contract, metrics, CLI and Python |
| [Leaderboard submissions](leaderboard.md) | Test-input access, `browsesafe predict`, file format, limits, publication |
| [Adaptive attacks](adaptive.md) | Contexts, seeds, policies, budgets, judges, reports; worked example |
| [Providers](providers.md) | OpenAI-compatible endpoints and environment variables |
| [API reference](api.md) | Public classes and functions |
| [Contributing](contributing.md) | Development setup and conventions |
| [Changelog](../CHANGELOG.md) | Release history |

## Concepts in one minute

- A **pair** is two pages sharing a five-digit index: `00042_malicious` (label `yes`) and `00042_benign` (label `no`). The benign page is the control: the same page with the injection removed.
- A **view** is the representation a model sees: `html`, `markdown`, `accessibility` or `screenshot`.
- A **benchmark** is a seeded sample of pairs plus a **manifest** that records the source, revision, filters, selected pair ids and every exclusion.
- A **detector** maps a `DetectorInput` (content and view only; no labels) to a `Prediction`.
- A **campaign** attacks one **context** (page, task, goal, insertion slot) within a **budget**, judged by your **judge**.
