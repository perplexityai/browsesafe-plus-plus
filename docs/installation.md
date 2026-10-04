# Installation

Python 3.10 or later. The package is not on PyPI yet; install it from GitHub:

```bash
pip install "browsesafe-pp @ git+https://github.com/perplexityai/browsesafe-plus-plus.git"               # core: loading and scoring
pip install "browsesafe-pp[eval] @ git+https://github.com/perplexityai/browsesafe-plus-plus.git"         # + httpx for the OpenAI-compatible provider
pip install "browsesafe-pp[screenshot] @ git+https://github.com/perplexityai/browsesafe-plus-plus.git"   # + Pillow for the screenshot view
pip install "browsesafe-pp[browser] @ git+https://github.com/perplexityai/browsesafe-plus-plus.git"      # + Playwright (and httpx) for the browser target
python -m playwright install chromium             # once, for the browser target
```

From a source checkout:

```bash
git clone https://github.com/perplexityai/browsesafe-plus-plus.git
cd browsesafe-plus-plus
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

| Extra | Adds | Needed for |
|---|---|---|
| `eval` | `httpx` | `OpenAICompatible`, `ModelDetector`, `browsesafe evaluate --model` |
| `screenshot` | `Pillow` | `view="screenshot"` and `browsesafe_pp.screenshot` |
| `browser` | `playwright`, `httpx` | `browsesafe_pp.browser.PlaywrightBackend`, `ChatModel` |
| `dev` | `pytest`, `httpx`, `Pillow` | running the test suite |

`adaptive.LinearUCB` additionally needs NumPy (`pip install numpy`); the other policies are pure Python.

## Offline fixture

The SDK includes a generator for a tiny synthetic dataset with the published schema. Everything in the docs and examples runs against it without network access:

```bash
python -m browsesafe_pp.testing ./fixture --pairs 12
browsesafe info --source ./fixture
```

```python
from browsesafe_pp.testing import write_fixture
write_fixture("./fixture", n_pairs=12)
```

The fixture pages are fictional and their "injections" are harmless marker sentences.

## The real dataset

The SDK defaults to [perplexity-ai/browsesafe-plus-plus](https://huggingface.co/datasets/perplexity-ai/browsesafe-plus-plus), which holds the training split (29,500 pairs). The test split (9,175 pairs) is reserved for the [leaderboard](https://browsesafe.perplexity.ai/leaderboard/), and a submission endpoint for it is coming soon. To select another repository, set:

```bash
export BROWSESAFE_DATASET=perplexity-ai/browsesafe-plus-plus
```

or download the shards and use `source="/path/to/export"`. The first Hub load downloads all `data/members-*.parquet` shards to the standard Hugging Face cache; later loads reuse them (`offline=True` or `HF_HUB_OFFLINE=1` skips the network entirely).
