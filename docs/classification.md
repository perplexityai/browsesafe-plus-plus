# Classification: prompt-injection detection

The detection task: given one page in one view, decide whether it contains a prompt injection. Every attacked page is scored together with its control, so a detector that flags a topic, a site or a language rather than the injection pays for it in false positives on the paired controls.

## The detector contract

```python
import browsesafe_pp as bs

def detector(request: bs.DetectorInput) -> bs.Prediction | bool | None:
    request.content   # the page in the selected view (for screenshots: the image path)
    request.view      # "html", "markdown", "accessibility" or "screenshot"
    return bs.Prediction(injected=True, score=0.93)   # score is optional, in [0, 1]
```

- Return `Prediction(injected=True|False)`, a plain `bool`, or `None` / `Prediction(injected=None)` to **abstain**.
- An exception counts as an **error**; any other return value (or a score outside [0, 1]) is an **invalid prediction**. Neither becomes a label.
- The input carries no labels, annotations, ids or task text. Each page gets exactly one call; there are no retries.

```python
benchmark = bs.load(split="train", view="html", n_pairs=1000)
report = bs.evaluate(benchmark, detector)                                    # synchronous
report = await bs.evaluate_async(benchmark, async_detector, concurrency=16)   # overlapping requests
report.save("runs/my-detector")    # manifest.json, metrics.json, predictions.jsonl (no page text)
```

`evaluate_async` accepts sync or async callables and keeps the benchmark order in `report.predictions`.

`DetectorInput` and `Prediction` are also the wire format for leaderboard evaluation: `browsesafe serve` exposes your `predict` function as `POST /predict`, and `EndpointDetector` is the client SII uses to call it against the private test split. See [Serving a detector](#serving-a-detector).

## The BrowseSafe 1.0 prompt and `ModelDetector`

`bs.ModelDetector(provider)` turns any OpenAI-compatible model into a detector using the BrowseSafe 1.0 instruction prompt for general-purpose models (BrowseSafe paper, Appendix G.2):

- The prompt is the system message; the page (text or screenshot segments) is the user message.
- The model must call the tool `record_classification` with `{"classification": 0 | 1, "explanation": "..."}`. A JSON object in the message text is accepted as a fallback.
- `classification` must be the integer 0 or 1 and `explanation` a nonempty string; anything else is an `invalid_prediction`.

```python
from browsesafe_pp.evaluation import get_classification_prompt

prompt = get_classification_prompt("browsesafe-1.0")
print(prompt.text)               # full prompt text
prompt.response_schema           # JSON schema of the tool arguments
prompt.parse({"classification": 1, "explanation": "redirects data"})   # -> "yes"
prompt.metadata()["sha256"]      # record this with your results
```

> **Note on the published prompt.** The paper prints the prompt with a literal `...` where its step-by-step reasoning list would be (after "When analyzing content, think through these steps:"). The full list is not published, so the SDK keeps the `...` verbatim rather than inventing steps. Results with this prompt are comparable with each other and with runs that use the same published text.

You can pass your own `ClassificationPrompt(name, text, output_format="label")` (expects `{"label": "yes" | "no" | "abstain"}`) or `output_format="classification"`.

```python
detector = bs.ModelDetector(provider, prompt=my_prompt, max_tokens=4096, temperature=0.0, use_tools=True)
```

Set `use_tools=False` for servers without tool calling; the model must then answer with the JSON object.

## Serving a detector

Leaderboard results are produced by SII running your detector on the private `test` split. Instead of sending code or weights, you expose the detector over HTTP:

```bash
browsesafe serve --predict my_detector:predict --host 0.0.0.0 --port 8000
```

| Route | Request | Response |
|---|---|---|
| `POST /predict` | `{"content": str, "view": str, "task": str \| null}` | `{"injected": bool \| null, "score": float \| null}` |
| `GET /health` | | `{"status": "ok", "sdk_version": "0.2.0"}` |

The server resolves `module:function` and normalises return values exactly like `browsesafe evaluate`: a `bool`, `None`, or `Prediction` all work, and an async function is awaited per request. An exception inside `predict` answers `500`; a return value that is not a valid prediction answers `422`. Malformed requests get `400`.

`bs.EndpointDetector` is the matching client. It is a synchronous detector, so it drops into `bs.evaluate` like any other function:

```python
detector = bs.EndpointDetector("https://detector.example.com/predict", token="...", timeout=60, retries=2)
report = bs.evaluate(bs.load(split="train", n_pairs=50), detector)
detector.configuration        # endpoint, timeout, retries; never the token
detector.close()
```

Timeouts and 5xx responses are retried up to `retries` times with doubling backoff (`backoff=0.5` seconds); any other failure is an `error` outcome, `422` or an unparseable body is an `invalid_prediction`. Requests carry `Authorization: Bearer <token>` when a token is set. `EndpointDetector` needs `httpx` (`pip install "browsesafe-pp[eval]"`); `browsesafe serve` uses only the standard library.

## Metrics

Let TP, FP, TN, FN count **valid** decisions (not errors, abstentions or invalid predictions) over all pages: malicious pages are positives, controls are negatives.

| Metric | Definition |
|---|---|
| `precision_valid` | TP / (TP + FP) |
| `recall_valid` | TP / (TP + FN) |
| `f1_valid` | 2 TP / (2 TP + FP + FN) |
| `accuracy_valid` | (TP + TN) / valid |
| `balanced_accuracy_valid` | (recall + TN / (TN + FP)) / 2 |
| `benign_fpr_valid` | FP / (FP + TN): false alarms on controls |
| `coverage` | valid / pages |
| `attack_recall_all` | TP / pairs: failures on attacked pages count as misses |
| `pair_correctness_all` | pairs with both members valid and correct / pairs |
| `paired_outcomes` | counts of `both_correct`, `attack_only_correct`, `control_only_correct`, `neither_correct`, `incomplete_outcomes` |
| `status_by_arm` | valid / abstained / error / invalid counts for malicious and benign pages |

A ratio is `null` when its denominator is zero. Report `coverage` next to any `*_valid` metric: a detector that abstains on hard pages can look accurate on the rest. `pair_correctness_all` is the strictest single number: it needs the injection found **and** its control left alone.

`report.predictions` holds one record per page: `entry_id`, `pair_id`, `expected_injected`, `injected`, `score`, `status`. Use `browsesafe_pp.evaluation.score(benchmark, outcomes)` to rescore saved `(status, injected, score)` tuples.

## Command line

```bash
# Your own detector (sync or async function), on 200 train pairs
browsesafe evaluate --predict my_detector:predict --n-pairs 200 --output runs/mine

# A model through the provider
export BROWSESAFE_MODEL_ENDPOINT=https://api.openai.com/v1 BROWSESAFE_API_KEY=sk-...
browsesafe evaluate --model gpt-4o-mini --view accessibility --n-pairs 0 --concurrency 32 --output runs/full

# Offline, against the fixture
python -m browsesafe_pp.testing ./fixture
browsesafe evaluate --source ./fixture --predict examples.detection_baseline:predict
```

| Option | Default | Meaning |
|---|---|---|
| `--source` / `--dataset` / `--revision` / `--offline` | Hub, `$BROWSESAFE_DATASET`, `main` | Where to load from |
| `--split` | `train` | Partition, or `all`. `test` is private and only loads from a source that contains it ([Leaderboard](leaderboard.md)) |
| `--view` | `html` | `html`, `markdown`, `accessibility`, `screenshot` |
| `--n-pairs` | `100` | Number of pairs; `0` for all |
| `--seed` | `42` | Sampling seed |
| `--predict` / `--model` | (one required) | Detector `module:function`, or model name |
| `--endpoint` | `$BROWSESAFE_MODEL_ENDPOINT` | Provider base URL |
| `--concurrency` | `8` | Requests in flight |
| `--max-tokens` | `4096` | Output token limit per classification |
| `--evaluator` | `self` | Recorded in `manifest["evaluator"]`; SII writes `sii` for verified runs |
| `--run-id` | none | Recorded in `manifest["run_id"]` |
| `--output` | none | New directory for the report |

```text
browsesafe serve --predict MODULE:FUNCTION [--host 127.0.0.1] [--port 8000]
```

## Reporting results

Record the dataset revision (`report.manifest["revision"]`), split, view, number of pairs, the prompt hash (`report.manifest["detector"]["prompt"]["sha256"]` for `ModelDetector`), the model and its settings. Compare detectors only on the same selection (`report.manifest["selection_digest"]`).

Every report manifest also carries provenance: `partition` (the evaluated split, or `"all"`), `evaluator` (`"self"` by default; `"sii"` for verified leaderboard runs) and `run_id` (`None` unless the evaluator assigns one). Pass `evaluator=` and `run_id=` to `evaluate`, `evaluate_async` or `score` (or `--evaluator` / `--run-id` on the command line) to set them. Results on `train` are self-reported; only SII-run results on the private `test` split appear on the [leaderboard](leaderboard.md).
