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

The fixture pages are fictional and their "injections" are harmless marker sentences. `python -m browsesafe_pp.testing ./test-fixture --test-inputs` (or `write_test_fixture`) writes the same pages in the label-free layout of the leaderboard's test inputs, plus a `labels.parquet` for local checks.

## Dataset access

The SDK defaults to [perplexity-ai/browsesafe-plus-plus](https://huggingface.co/datasets/perplexity-ai/browsesafe-plus-plus). It has one configuration with a `train` split (29,500 labelled pairs) and a `test` split, the label-free inputs for the [leaderboard](leaderboard.md) (one page from each of 9,175 pairs, under `test/`). The dataset is gated, and one request covers both:

1. Log in to Hugging Face, open the dataset page, fill in the request form (name, affiliation, intended use) and accept the terms. Access is granted automatically.
2. Authenticate on the machine that loads the data: `hf auth login`, the `HF_TOKEN` environment variable, or `token=` in Python.

Without access, a Hub load raises `PermissionError` with these steps. The terms are in `LICENSE.md` in the dataset repository: the training split is released under CC BY 4.0, and the test inputs are for evaluation only ([summary](leaderboard.md#1-access-and-licence)).

Each load downloads only its split's shards to the standard Hugging Face cache: `data/members-*.parquet` for `bs.load`, `test/data/test-*.parquet` for `bs.load_test_inputs`. Later loads reuse them (`offline=True` or `HF_HUB_OFFLINE=1` skips the network entirely). To read another repository, set `BROWSESAFE_DATASET=<id>`, or download the files and use `source="/path/to/export"`.
