<div align="center">

<img src="docs/assets/wordmark.svg" alt="BrowseSafe++" width="329">

[Documentation](docs/index.md) · [Benchmark](https://huggingface.co/datasets/perplexity-ai/browsesafe-plus-plus) · [Classification](docs/classification.md) · [Adaptive attacks](docs/adaptive.md) · [Leaderboard](https://browsesafe.perplexity.ai/leaderboard/)

</div>

BrowseSafe++ is a benchmark for detecting indirect prompt injections in web pages. Each entry pairs an injected page with a control page that differs only by the injection.

The SDK lets you:

- Load attack/control pairs as HTML, Markdown, accessibility trees, or screenshots.
- Evaluate a detector in Python or from the command line.
- Run adaptive attack campaigns against an agent.
- Serve a detector for verified leaderboard evaluation.

## Install

```bash
pip install "browsesafe-pp[eval]"
```

Requires Python 3.10 or later. Use `[eval,screenshot]` for screenshot evaluation or `[browser]` for the Playwright adapter. From a checkout, run `pip install -e ".[dev]"`. See [installation](docs/installation.md).

## Quick start

Create a local sample dataset:

```bash
python -m browsesafe_pp.testing ./fixture
browsesafe info --source ./fixture
```

```python
import browsesafe_pp as bs

benchmark = bs.load(source="./fixture", split="train", view="html")
pair = benchmark.pairs[0]
print(pair.malicious.injected, pair.benign.injected)   # True False
print(pair.malicious.evidence)                         # the injection text, as it appears in the page

def my_detector(request: bs.DetectorInput) -> bs.Prediction:
    return bs.Prediction(injected="assistant note" in request.content.lower())

report = bs.evaluate(benchmark, my_detector)
print(report.metrics["f1_valid"], report.metrics["pair_correctness_all"])
```

For the full benchmark, omit `source=`. The SDK loads [perplexity-ai/browsesafe-plus-plus](https://huggingface.co/datasets/perplexity-ai/browsesafe-plus-plus) by default. You can also pass a local directory of Parquet shards. See [dataset loading](docs/data.md).

The public dataset contains the `train` partition only. The `test` partition is private: SII runs it against your detector or agent endpoint and publishes the result on the [leaderboard](docs/leaderboard.md).

## Dataset

| | |
|---|---|
| Size | **38,675 attack/control pairs (77,350 pages)** |
| Splits | **train 29,500 pairs (public) / test 9,175 pairs (private, leaderboard only)**; both pages of a pair share a split |
| Views | HTML, Markdown, accessibility tree, full-page screenshot |
| Taxonomy | **12 domains, 24 attack types, 18 placements, 6 attacker goals**, plus page register, linguistic style and risk category |
| Languages | **80 languages** (ISO 639-1 codes) across page text and injection text |
| Per pair | injected page, control page, injection fragments, benign user task, expected outcome, attacker goal, whether the goal needs action tools, detection date |

Field definitions are in [docs/data.md](docs/data.md). Set `BROWSESAFE_DATASET` to override the repository, or use a local `source=` directory. Use `revision=` to pin a dataset version.

## Load screenshots

```bash
pip install "browsesafe-pp[screenshot]"
```

```python
import browsesafe_pp as bs
from PIL import Image

benchmark = bs.load(split="train", view="screenshot", n_pairs=20)
pair = benchmark.pairs[0]
attack_image = Image.open(pair.malicious.content)
control_image = Image.open(pair.benign.content)
```

For this view, `Page.content` is a local JPEG path. The first Hub load downloads the full screenshot collection (about 60 GB), even when `n_pairs` is small; later loads reuse the cache. See [screenshots](docs/data.md#screenshots) to download a single image or use a local copy. Model evaluation uses up to eight 1,536-pixel image segments per page and reports any omitted height.

## Evaluate a model

```bash
export BROWSESAFE_MODEL_ENDPOINT=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export BROWSESAFE_API_KEY=sk-...
browsesafe evaluate --model gpt-4o-mini --split train --view markdown --n-pairs 200 --output runs/gpt4o-mini
```

```python
import asyncio
import browsesafe_pp as bs

async def main():
    benchmark = bs.load(split="train", view="html", n_pairs=200)
    provider = bs.OpenAICompatible()                  # reads the BROWSESAFE_* variables
    detector = bs.ModelDetector(provider)             # BrowseSafe 1.0 classification prompt
    report = await bs.evaluate_async(benchmark, detector, concurrency=16)
    await provider.aclose()
    report.save("runs/my-model")
    return report.metrics

print(asyncio.run(main()))
```

For a custom detector, use `browsesafe evaluate --predict my_module:predict`. Metric definitions are in [docs/classification.md](docs/classification.md).

## Get on the leaderboard

Leaderboard results are measured on the private `test` split by SII. Expose your detector with the `/predict` contract and send the URL through the [submission form](https://browsesafe.perplexity.ai/leaderboard/#submit):

```bash
browsesafe serve --predict my_module:predict --host 0.0.0.0 --port 8000
```

```python
detector = bs.EndpointDetector("https://detector.example.com/predict", token="...")   # what SII calls
report = bs.evaluate(bs.load(split="train", n_pairs=50), detector)                    # check your deployment
```

Agents are evaluated through an OpenAI-compatible chat-completions endpoint with tool calling. See [docs/leaderboard.md](docs/leaderboard.md) for both tracks and why the test split is not distributed.

## Run an adaptive campaign

A campaign evaluates candidate injections against an agent within a specified budget. It runs a clean baseline, tests each candidate on a fresh target, and repeats successful attacks for confirmation.

The example below assumes you have defined `page_text`, `my_attacker`, `MyAgentTarget`, and `my_judge`. For a runnable example, see [adaptive_local.py](examples/adaptive_local.py).

```python
import asyncio
from browsesafe_pp import adaptive as a

context = a.Context(context_id="demo", task="What are the opening hours?",
                    observation=page_text, goal="Make the answer end with DEMO-MARKER",
                    slot=a.Slot(len(page_text), len(page_text)))   # where payloads are inserted

result = asyncio.run(a.run(
    context, attacker=my_attacker, target_factory=MyAgentTarget, judge=my_judge,
    mode="generate", confirmations=2, output_dir="runs/demo",
    budget=a.Budget(max_candidates=16, max_attacker_calls=16, max_target_calls=40, max_judge_calls=40),
))
print(result.summary["confirmed_success"], result.summary["stop_reason"])
```

Use `a.seeds.from_benchmark(...)` to seed a campaign from the training split and `a.reports.aggregate(...)` to combine results. See [adaptive_seeded.py](examples/adaptive_seeded.py) and the [adaptive API guide](docs/adaptive.md).

## Examples

| Script | What it shows |
|---|---|
| [`examples/detection_baseline.py`](examples/detection_baseline.py) | Keyword baseline scored on pairs |
| [`examples/evaluate_model.py`](examples/evaluate_model.py) | Model evaluation through the provider (`--mock` runs offline) |
| [`examples/adaptive_local.py`](examples/adaptive_local.py) | A full campaign against a local stand-in agent |
| [`examples/adaptive_seeded.py`](examples/adaptive_seeded.py) | Seeded campaigns over held-out contexts with portfolio search |

Each accepts `--source ./fixture` to run against the bundled synthetic data.

## Documentation

[Overview](docs/index.md) · [Installation](docs/installation.md) · [Data](docs/data.md) · [Classification](docs/classification.md) · [Adaptive attacks](docs/adaptive.md) · [Leaderboard](docs/leaderboard.md) · [Providers](docs/providers.md) · [API reference](docs/api.md) · [Contributing](docs/contributing.md) · [Changelog](CHANGELOG.md)

## Leaderboard

See the [leaderboard](https://browsesafe.perplexity.ai/leaderboard/) for detection and adaptive-attack results on the private test split, and [docs/leaderboard.md](docs/leaderboard.md) for how to submit.

## Citation

```bibtex
TBD
```

BrowseSafe++ builds on BrowseSafe ([arXiv:2511.20597](https://arxiv.org/abs/2511.20597)); please cite it as well when you use the BrowseSafe 1.0 prompt or data.

## License

See the [dataset card](https://huggingface.co/datasets/perplexity-ai/browsesafe-plus-plus) for the dataset license.
