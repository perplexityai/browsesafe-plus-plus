# Submitting to the leaderboard

The test split (9,175 attack/control pairs) is scored by the [BrowseSafe++ leaderboard Space](https://huggingface.co/spaces/perplexity-ai/browsesafe-plus-plus-leaderboard). The test inputs hold one page of every pair, the attacked page or its control, so there are 9,175 pages and half of them carry an injection. You download them without labels, run your detector on your own hardware, and upload a predictions file. The Space returns aggregate scores for each view and, if you ask it to, publishes them on the [leaderboard](https://browsesafe.perplexity.ai/leaderboard/) next to the paper's models.

## 1. Access and licence

The test pages are the `test` split of the gated BrowseSafe++ dataset, [perplexity-ai/browsesafe-plus-plus](https://huggingface.co/datasets/perplexity-ai/browsesafe-plus-plus), in the same configuration as the training split (files under `test/`). If you can load the training split, you already have access; otherwise request it once and log in as described in [Dataset access](installation.md#dataset-access).

The test inputs are under an evaluation-only licence (the training split is CC BY 4.0). In short:

- Do not train or fine-tune on the test pages.
- Do not redistribute or publish test content.
- Report results together with the dataset revision that `browsesafe predict` prints.

The full text is in `LICENSE.md` in the dataset repository.

Each test row is one page in the training split's columns: an opaque `entry_id` (`TestPage.id`, the `id` of your predictions), the `benign_task` and the four views, `content_html`, `content_markdown`, `content_ax_yaml` and the `screenshot` image (`test/screenshots/<entry_id[:2]>/<entry_id>.jpg`). Labels, arm, title and every annotation are empty, and no page's twin is among the rows. `datasets.load_dataset("perplexity-ai/browsesafe-plus-plus", split="test")` reads the same rows.

## 2. Run your detector

```bash
browsesafe predict --predict my_detector:predict --view html --view markdown --output predictions.jsonl
```

`--predict` names a sync or async function, or a callable object, that follows the [detector contract](classification.md#the-detector-contract). Repeat `--view` for each view you want scored. The command loads each view, calls the detector once per page and writes one file for all the views. It then prints, for each view, how many pages it read, how many records it wrote, how many were abstentions and how many pages it left out because of an exception or an invalid return.

| Option | Default | Meaning |
|---|---|---|
| `--predict` | (required) | Detector as `module:function` |
| `--view` | (required, repeatable) | `html`, `markdown`, `accessibility`, `screenshot` |
| `--output` | (required) | New JSON Lines file; an existing file is never overwritten |
| `--source` / `--dataset` / `--revision` / `--offline` | Hub, `$BROWSESAFE_DATASET` or `perplexity-ai/browsesafe-plus-plus`, `main` | Where to load the test inputs from; only files under `test/` are downloaded |
| `--limit` | all pages | Only the first N pages of each view, for a trial run |
| `--concurrency` | `8` | Calls in flight for an async detector |

To score a model through an [OpenAI-compatible provider](providers.md), put the detector in a module and point `--predict` at it:

```python
# my_model.py
import browsesafe_pp as bs

detector = bs.ModelDetector(bs.OpenAICompatible())   # reads the BROWSESAFE_* variables
```

```bash
browsesafe predict --predict my_model:detector --view accessibility --concurrency 16 --output predictions.jsonl
```

The same steps in Python:

```python
import browsesafe_pp as bs

records = []
for view in ("html", "markdown", "accessibility"):
    pages = bs.load_test_inputs(view=view)            # tuple of TestPage(id, content, view, task)
    records += bs.predict_test(pages, my_detector)    # async detectors: add concurrency=16
bs.write_predictions("predictions.jsonl", records)
```

`page.model_input` is task-free, as in local evaluation; `page.task` holds the benign task if your detector uses one. Inside a running event loop (for example a notebook), use `await bs.predict_test_async(pages, detector, concurrency=16)`. For `view="screenshot"`, `page.content` is a local JPEG path.

To try the flow offline, write a small fixture with the same layout and use it as the source:

```bash
python -m browsesafe_pp.testing ./test-fixture --test-inputs
browsesafe predict --source ./test-fixture --predict examples.detection_baseline:predict --view html --output trial.jsonl
```

## 3. The predictions file

The file is JSON Lines, one record per page and view:

```json
{"id": "3f9c0a1b2d4e", "view": "html", "injected": true, "score": 0.91}
{"id": "3f9c0a1b2d4e", "view": "markdown", "injected": null}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string | The page's `entry_id` in the test inputs (`TestPage.id`) |
| `view` | string | `html`, `markdown`, `accessibility` or `screenshot` |
| `injected` | `true`, `false` or `null` | The decision; `null` abstains |
| `score` | number, optional | Detector score in [0, 1] |

A file may cover any subset of the four views and counts as one submission. For each view in the file, a test page without a record counts as an error: it lowers `coverage` and `attack_recall_all`, exactly as in local evaluation. If the file repeats an (`id`, `view`) pair, names an unknown `id` or view, or contains malformed values, it is rejected with a message. A rejected file does not use up your daily submission. `bs.write_predictions` runs the same checks before it writes the file.

## 4. Submit

Open the [leaderboard Space](https://huggingface.co/spaces/perplexity-ai/browsesafe-plus-plus-leaderboard), log in with Hugging Face and fill in the form. Upload the file and enter a model name (required). You can also add an organisation, a link to code or a paper, and a short description. Tick "Request publication on the leaderboard" if you want the result shown there; the maintainers review every request before it appears.

| Limit | Rule |
|---|---|
| Account | A Hugging Face account at least 60 days old |
| Rate | One scored submission per account per 7 days |

For each view, the Space returns F1, accuracy, false-positive rate, precision, recall, balanced accuracy, coverage and attack recall over all attacked pages (`f1_valid`, `accuracy_valid`, `benign_fpr_valid`, `precision_valid`, `recall_valid`, `balanced_accuracy_valid`, `coverage`, `attack_recall_all`), with the definitions in [Classification](classification.md#metrics). Pair metrics such as `pair_correctness_all` need both members of a pair and are not computed on the test inputs. The Space returns aggregate scores only, never per-page results.

## 5. Publication

- Publication is reviewed by the maintainers; nothing is published automatically. An approved request appears on the leaderboard Space within minutes and on the website within an hour.
- A view can be requested only if every one of its pages has a decision: coverage 100%, with no abstentions and no missing pages. The other views are scored privately, and the Space gives the reason.
- The leaderboard keeps one row per account and model name. A newer approved request replaces the older row.
- For each view, the leaderboard shows the five best community results by F1 and hides results with F1 below 50 by default; hidden rows stay one click away.
- A row shows the views it covers (F1, accuracy and FPR per view). The text board's Average appears only when all three text views are present; screenshot results appear on the screenshot board.
- Every result records the test version (`v1`). Published results are not re-scored when a new test version appears.
