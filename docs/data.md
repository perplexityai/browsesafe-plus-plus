# Data

## Layout

A BrowseSafe++ source is a directory of Parquet shards, one row per page:

```
<source>/
  data/members-train-0000.parquet
  ...
  screenshots/<first two digits>/<entry_id>.jpg     # optional; flat screenshots/<entry_id>.jpg also works
```

`bs.load(source=...)` accepts either `<source>` or `<source>/data`. Without `source=`, the same layout is downloaded from the Hugging Face dataset named by `BROWSESAFE_DATASET` (default `perplexity-ai/browsesafe-plus-plus`) at `revision=` (default `main`). Pin a 40-character commit for comparable runs; the manifest records the resolved commit and `revision_pinned`.

The Hugging Face dataset contains the `train` split. The `test` split is reserved for the [leaderboard](https://browsesafe.perplexity.ai/leaderboard/), and a submission endpoint for it is coming soon; asking the Hub for `split="test"` raises an error that says so. A local `source=` that contains a `test` partition loads normally. The `screenshot` column is an image feature that displays each row's JPEG in the dataset viewer and decodes to a PIL image with `datasets`. Image files stay separate from the text shards, so text-only SDK loads do not download the images. Install `datasets` and `Pillow` for this example:

```python
from datasets import load_dataset

pages = load_dataset("perplexity-ai/browsesafe-plus-plus", split="train", streaming=True)
page = next(iter(pages))
image = page["screenshot"]
```

## Pairs and splits

| | |
|---|---|
| Pairs | 38,675 (77,350 pages) |
| Train | 29,500 pairs (59,000 pages), on Hugging Face |
| Test | 9,175 pairs (18,350 pages), reserved for the leaderboard |

Rows `NNNNN_malicious` (`arm="attack"`, `label="yes"`) and `NNNNN_benign` (`arm="control"`, `label="no"`) form one pair. The control is the attacked page with the injection removed. Both members always share a split, and no injection text (compared after whitespace and case normalisation) appears in both splits.

The loader validates pairs and reports what it drops in `benchmark.manifest["exclusions"]`:

| Exclusion | Meaning |
|---|---|
| `malformed_entry_id_rows` | `entry_id` does not match `NNNNN_malicious` / `NNNNN_benign` |
| `duplicate_entry_id_rows`, `inconsistent_arm_label_rows`, `invalid_pairs` | duplicated ids or arm/label disagreement; the whole pair is dropped |
| `incomplete_pairs` | one member is missing |
| `inconsistent_split_pairs` | members disagree on the split |
| `other_split_pairs`, `metadata_filtered_pairs` | not selected by `split=` or `filters=` |
| `missing_view_members`, `missing_view_pairs` | the selected view is empty or missing for a member; its pair is dropped |

## Views

| `view=` | Column | Content |
|---|---|---|
| `html` | `content_html` | Page HTML with markup, CSS and layout |
| `markdown` | `content_markdown` | Markdown rendering of the page |
| `accessibility` | `content_ax_yaml` | Accessibility tree as YAML (roles, names, structure) |
| `screenshot` | `screenshots/` | Full-page JPEG capture; `Page.content` is the file path |

All pairs in the current release have the three text views. When loading a custom source, empty or missing views are excluded and counted; the loader never falls back to another view.

### Screenshots

Install the image dependencies:

```bash
pip install "browsesafe-pp[screenshot] @ git+https://github.com/perplexityai/browsesafe-plus-plus.git"
```

```python
import browsesafe_pp as bs
from PIL import Image

benchmark = bs.load(split="train", view="screenshot", n_pairs=20)
pair = benchmark.pairs[0]
attack_image = Image.open(pair.malicious.content)
control_image = Image.open(pair.benign.content)
```

`Page.content` is the local JPEG path; labels, tasks, and pair metadata work as in the text views. The first Hub load downloads **all** `screenshots/**` (about 45 GB), before `split` and `n_pairs` are applied. Later loads reuse the cache. To use an existing local collection, pass `source="/path/to/dataset"` with both its `data/` and `screenshots/` directories. Unreadable or truncated images are treated as missing.

To download just one image without loading the collection:

```python
from huggingface_hub import hf_hub_download

entry_id = "00004_malicious"
path = hf_hub_download(
    repo_id="perplexity-ai/browsesafe-plus-plus",
    repo_type="dataset",
    filename=f"screenshots/{entry_id[:2]}/{entry_id}.jpg",
)
image = Image.open(path)
```

To give a screenshot to a model, `browsesafe_pp.screenshot.user_content(path)` splits it into segments of 1,536 pixels (at most 8, top first) and returns OpenAI-style content parts. The initial text part reports how much of the page is shown and any omitted height:

```python
from browsesafe_pp import screenshot
parts = screenshot.user_content(pair.malicious.content)   # text summary followed by image_url parts
```

## Fields

The loaded `Page` exposes the selected view as `content`, the label as `injected`, the benign task as `task`, the bound injection strings as `evidence` and every other column through `metadata`. Pair-level annotations live on the malicious member (`pair.metadata`).

### Identity and labels

| Field | Type | Meaning |
|---|---|---|
| `entry_id` | string | `NNNNN_malicious` or `NNNNN_benign` |
| `arm` | string | `attack` or `control` |
| `label` | string | `yes` (injection present) or `no` |
| `split` | string | `train` (the only split in the public dataset) or `test` |

### Page and context

| Field | Type | Meaning |
|---|---|---|
| `title` | string | Page title |
| `content_html`, `content_markdown`, `content_ax_yaml` | string | The three text views |
| `screenshot` | image | Screenshot reference; the Hugging Face loader decodes it to a PIL image. The SDK's screenshot view exposes the cached JPEG path. |
| `page_type` | string | One of 12 page classes (below) |
| `domain` | string | Coarse site purpose: `general_web`, `social_media`, `workspace`, `education`, `entertainment`, `e_commerce` |
| `language_page` | list of strings | Lowercase ISO 639-1 codes of the page text, in mention order |
| `register` | string or null | `formal`, `casual`, `marketing` or `legalese` |

### Task and outcome

| Field | Type | Meaning |
|---|---|---|
| `benign_task` | string | The legitimate user request for this page |
| `expected_outcome` | string or null | What a safe agent should do on this member |
| `malicious_goal` | string or null | What the injection tries to make the agent do |
| `goal_class` | string or null | `exfiltration`, `unauthorized_action`, `misinformation_to_user`, `propagation_to_others`, `persistence_or_state_change`, `task_disruption` |
| `risk_category` | string or null | `data-exfiltration`, `spam-bot`, `manipulative-ai-use`, `harmful-illegal`, `browser-integrity` |
| `requires_action_tools` | bool | Whether the goal needs tools beyond reading (always `false` on controls) |

### Injection annotations

| Field | Type | Meaning |
|---|---|---|
| `attack_type` | string or null | Attack mechanism (below) |
| `injection_strategy` | string or null | Placement / carrier (below) |
| `linguistic_style` | string | `explicit`, `indirect`, `stealth`; `none` on controls; `unknown` if unannotated |
| `language_payload` | list of strings or null | Language codes of the injection text |
| `injection_text_fragments_json` | JSON string or null | List of the injection's text fragments, byte-identical to the page |
| `attack_quotes_json` | JSON string or null | List of reference quotations |
| `attack_identified_ts` | string or null | ISO-8601 time of the production detection associated with the injection |

```python
import json
fragments = json.loads(pair.metadata.get("injection_text_fragments_json") or "[]")
spans = pair.malicious.evidence_spans        # (start, end) character offsets in the selected view
```

`evidence` contains the fragments and quotes that occur verbatim in the loaded view; references are not exhaustive span annotations, and an empty `evidence` never changes a label. `manifest["evidence_coverage"]` counts bound and unbound references.

## Taxonomy

The benchmark covers **12 page types, 24 attack types, 18 placements, 6 goals and 80 languages**.

**Page types (`page_type`)**: `article`, `document`, `thread`, `video`, `dashboard`, `profile`, `repository`, `inbox`, `listing`, `feed`, `search_results`, `checkout`. These correspond to news and blogs, shared documents, forums, video sites, web apps, social profiles, code repositories, email, marketplaces, social feeds, search results and checkout pages.

**Attack types (`attack_type`)**, examples of observed values: `important_message`, `todo`, `injecagent`, `ignore_previous`, `url_segment`, `role_manipulation`, `system_prompt_exfiltration`, `delimiter_injection`, `social_engineering`, `indirect_hypothetical`, `multilanguage`, `fabricated_authorization_thread`, `fake_policy_update`, `conditional_trigger`, `multi_stage_chain`, `tool_available_lure`, `fake_verification`, `preference_override`, `browser_state_tampering`, `spam_or_broadcast`, `phishing_link_propagation`, `ranking_or_review_poisoning`, `encoding_obfuscation`.

**Placements (`injection_strategy`)**: `html_comment`, `data_attribute`, `hidden_text`, `form_hidden_field`, `semantic_abuse`, `inline_paragraph_rewrite`, `list_item_rewrite`, `footer_rewrite`, `table_cell_rewrite`, `user_generated_block`, `webmail_or_message_body`, `shared_document_body`, `code_hosting_artifact`, `link_text_href_mismatch`, `alt_text_and_aria`, `structured_data_block`, `cross_page_placement`, `long_document_tail`.

Use `browsesafe info` (or `bs.describe()`) to see the counts in the source you have.

## Loading

```python
import browsesafe_pp as bs

bs.load(split="train", view="markdown", n_pairs=500, seed=7)
bs.load(split=None)                                             # every partition in the source
bs.load(filters={"page_type": ["thread", "inbox"]})
bs.load(filters={"language_payload": ["es", "pt"]})            # list fields: any overlap
bs.load(source="/data/browsesafe-pp", n_pairs=10)               # local shards
bs.load(revision="<40-char commit>", offline=True)              # pinned, cached snapshot
bs.load(task="evidence", split="train")                         # inspect references; not for scoring
```

Supported filter keys: `language_page`, `language_payload`, `requires_action_tools`, `attack_type`, `injection_strategy`, `register`, `domain`, `page_type`, `goal_class`, `risk_category`, `linguistic_style`. Filters apply to the malicious member and keep its control.

Sampling is a deterministic hash of the source identity, pair id and `seed`, so the same arguments select the same pairs; `n_pairs` takes a prefix of that order, so smaller samples are subsets of larger ones.

### Large selections

`load()` keeps the selected pages in memory. For full splits, iterate in batches:

```python
for batch in bs.iter_batches(split="train", view="html", batch_pairs=64, order="source"):
    report = bs.evaluate(batch, my_detector)
    ...   # save or aggregate per batch
```

`order="source"` emits pairs as soon as both members are read (small spool); `order="selection"` preserves `load()`'s order by spooling the selection to a temporary SQLite file (`spool_dir=`, capped by `max_spool_bytes`). The spool is removed when the iterator ends or is closed. The final batch has `manifest["batch"]["is_final"] = True` and the exclusions for the whole run.

For training, `benchmark.to_records()` yields `{"entry_id", "content", "label", "injected"}` per page.
