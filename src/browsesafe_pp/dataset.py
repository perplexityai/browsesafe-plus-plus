"""Paired loading of BrowseSafe++ pages.

A BrowseSafe++ source is a set of ``members-*.parquet`` shards with one row per
page. Two rows that share a five-digit group index form a pair: the
``<index>_malicious`` page carries a prompt injection and the ``<index>_benign``
page is the same page without it.

The loader reads in two passes. The first pass reads only the metadata columns
of every shard, validates the pairs and selects a reproducible sample. The
second pass materializes the selected view (HTML, Markdown, accessibility tree
or screenshot path) for the selected pages only, in small Arrow batches.
Screenshots embedded in the shards are written to a local cache file per page;
shards without embedded images use the source's screenshot files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

DATASET_ENV = "BROWSESAFE_DATASET"
DEFAULT_DATASET = "perplexity-ai/browsesafe-plus-plus"
DEFAULT_REVISION = "main"
DEFAULT_SPLIT = "train"
VIEWS = {"html": "content_html", "markdown": "content_markdown", "accessibility": "content_ax_yaml"}
SCREENSHOT_VIEW = "screenshot"
ALL_VIEWS = (*VIEWS, SCREENSHOT_VIEW)
SHARD_PATTERN = "members-*.parquet"
FILTERS = frozenset({
    "language_page", "language_payload", "requires_action_tools", "attack_type",
    "injection_strategy", "register", "domain", "page_type", "goal_class", "risk_category",
    "linguistic_style",
})
_ENTRY = re.compile(r"^(\d+)_(malicious|benign)$")
_SHA = re.compile(r"[0-9a-f]{40}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def dataset_id(dataset: str | None = None) -> str:
    """The Hugging Face dataset id: the argument, else ``$BROWSESAFE_DATASET``, else the default."""
    return dataset or os.environ.get(DATASET_ENV) or DEFAULT_DATASET


@dataclass(frozen=True)
class DetectorInput:
    """What a detector sees: the page content in one view. Labels and annotations are excluded."""

    content: str
    view: str
    task: str | None = None


@dataclass(frozen=True)
class Prediction:
    """A detector decision. ``injected=None`` abstains; ``score`` is an optional value in [0, 1]."""

    injected: bool | None
    score: float | None = None


@dataclass(frozen=True)
class Page:
    """One page of a pair in the selected view."""

    entry_id: str
    content: str
    view: str
    task: str | None
    injected: bool
    metadata: dict[str, Any]
    evidence: tuple[str, ...] = ()

    @property
    def model_input(self) -> DetectorInput:
        """The default, task-free detector input."""
        return DetectorInput(content=self.content, view=self.view)

    @property
    def evidence_spans(self) -> tuple[tuple[int, int], ...]:
        """Every occurrence of every evidence string, as half-open character offsets in ``content``."""
        spans: set[tuple[int, int]] = set()
        for quote in self.evidence:
            start = 0
            while (offset := self.content.find(quote, start)) >= 0:
                spans.add((offset, offset + len(quote)))
                start = offset + 1
        return tuple(sorted(spans))


@dataclass(frozen=True)
class Pair:
    """An injected page and its benign control."""

    pair_id: str
    malicious: Page
    benign: Page

    @property
    def metadata(self) -> dict[str, Any]:
        """Pair-level annotations (taxonomy, goal, detection time) live on the malicious member."""
        return self.malicious.metadata


@dataclass
class Benchmark:
    """A loaded selection of complete pairs plus a manifest describing how it was selected."""

    pairs: tuple[Pair, ...]
    manifest: dict[str, Any]

    def __iter__(self) -> Iterator[Page]:
        for pair in self.pairs:
            yield pair.malicious
            yield pair.benign

    def __len__(self) -> int:
        return 2 * len(self.pairs)

    def to_records(self) -> Iterator[dict[str, Any]]:
        """Supervised records (content plus label), one per page."""
        for page in self:
            yield {"entry_id": page.entry_id, "content": page.content,
                   "label": "yes" if page.injected else "no", "injected": page.injected}


def _hub_call(repo_id: str, function, /, **kwargs):
    """Call the Hub once; a refused access (gated, 401 or 403) becomes a PermissionError that says what to do."""
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    try:
        return function(**kwargs)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if not isinstance(exc, GatedRepoError) and status not in (401, 403):
            raise
        raise PermissionError(
            f"The Hugging Face Hub refused access to the gated dataset {repo_id}. Request access at "
            f"https://huggingface.co/datasets/{repo_id} (one request covers the training split and the test "
            "inputs), then log in with `hf auth login` or set HF_TOKEN.") from exc


def _resolve_revision(repo_id: str, revision: str, token: str | bool | None, offline: bool) -> str:
    """The commit of ``revision`` (unchanged when offline or already a commit)."""
    from huggingface_hub import HfApi, constants

    if offline or constants.HF_HUB_OFFLINE or _SHA.fullmatch(revision):
        return revision
    return _hub_call(repo_id, HfApi(token=token).dataset_info, repo_id=repo_id, revision=revision).sha


def _files(source: str | Path | None, dataset: str | None, revision: str, token: str | bool | None,
           offline: bool, subdir: str = "", pattern: str = SHARD_PATTERN) -> tuple[list[Path], dict[str, Any], str]:
    """Shards matching ``<subdir>/data/<pattern>`` (training: ``data/members-*``; test inputs: ``test/data/test-*``).

    Screenshot files are never downloaded here; see :func:`_screenshot_source`.
    """
    prefix = f"{subdir}/" if subdir else ""
    if source is None:
        from huggingface_hub import constants, snapshot_download

        repo_id = dataset_id(dataset)
        offline = offline or bool(constants.HF_HUB_OFFLINE)
        resolved = _resolve_revision(repo_id, revision, token, offline)
        root = Path(_hub_call(repo_id, snapshot_download, repo_id=repo_id, repo_type="dataset", revision=resolved,
                              token=token, local_files_only=offline, allow_patterns=[f"{prefix}data/{pattern}"]))
        commit = root.name if _SHA.fullmatch(root.name) else None
        source_info = {"source_kind": "huggingface", "dataset": repo_id, "requested_revision": revision,
                      "revision": commit, "revision_pinned": bool(_SHA.fullmatch(revision))}
        identity = commit or f"{repo_id}@{revision}"
    else:
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Local source is not a directory: {root}")
        source_info = {"source_kind": "local", "revision": None}
        identity = ""
    # A local source may be the dataset root, its <subdir>/ directory or the shard directory itself.
    data = next(path for path in (root / prefix / "data", root / "data", root) if path.is_dir())
    files = sorted(data.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {prefix}data/{pattern} shards found under {root}")
    if source is not None:
        hashes = []
        for path in files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes.append({"name": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
        identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        source_info.update({"fingerprint": identity, "files": hashes})
    source_info["shard_count"] = len(files)
    return files, source_info, identity


def _matches(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, requested in filters.items():
        choices = list(requested) if isinstance(requested, (list, tuple, set, frozenset)) else [requested]
        actual = row.get(key)
        if isinstance(actual, (list, tuple)):
            if not any(value in actual for value in choices):
                return False
        elif actual not in choices:
            return False
    return True


def _references(row: Mapping[str, Any], content: str) -> tuple[tuple[str, ...], dict[str, int]]:
    """Injection fragments and quotes that occur verbatim in ``content``."""
    refs: list[str] = []
    malformed = 0
    for key in ("injection_text_fragments_json", "attack_quotes_json"):
        raw = row.get(key)
        if raw is None:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            malformed += 1
            continue
        if not isinstance(parsed, list) or any(not isinstance(value, str) for value in parsed):
            malformed += 1
            continue
        refs.extend(value for value in parsed if value)
    distinct = tuple(dict.fromkeys(refs))
    bound = tuple(value for value in distinct if value in content)
    return bound, {"reference_strings": len(distinct), "bound_reference_strings": len(bound),
                   "unbound_reference_strings": len(distinct) - len(bound),
                   "malformed_reference_fields": malformed}


def _check_arguments(task: str, view: str, split: str | None, n_pairs: int | None, seed: int,
                     filters: Mapping[str, Any]) -> None:
    if task not in {"detection", "evidence"}:
        raise ValueError("task must be 'detection' or 'evidence'")
    if view not in ALL_VIEWS:
        raise ValueError(f"view must be one of {ALL_VIEWS}")
    if split is not None and not isinstance(split, str):
        raise ValueError("split must be a partition name or None")
    if n_pairs is not None and (type(n_pairs) is not int or n_pairs < 0):
        raise ValueError("n_pairs must be a nonnegative integer or None")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    unknown = set(filters) - FILTERS
    if unknown:
        raise ValueError("Unsupported filter names: " + ", ".join(sorted(unknown)))


def _prepare_load(*, task: str, split: str | None, split_column: str, view: str,
                  source: str | Path | None, dataset: str | None, revision: str,
                  token: str | bool | None, n_pairs: int | None, seed: int,
                  filters: Mapping[str, Any] | None, offline: bool):
    import pyarrow.parquet as pq

    filters = {key: sorted(value) if isinstance(value, (set, frozenset)) else value
               for key, value in dict(filters or {}).items()}
    _check_arguments(task, view, split, n_pairs, seed, filters)
    files, source_info, identity = _files(source, dataset, revision, token, offline)
    index: dict[str, dict[str, tuple[int, int, dict[str, Any]]]] = {}
    invalid_pairs: set[str] = set()
    exclusions: Counter[str] = Counter()
    seen_entries: set[str] = set()
    available_splits: set[str] = set()
    scanned = 0
    projection = sorted({"entry_id", "arm", "label", split_column, *filters})
    required = {"entry_id", "arm", "label", split_column} | ({VIEWS[view]} if view in VIEWS else set())
    for file_index, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        if required - columns:
            raise ValueError(f"{path.name} is missing required columns: " + ", ".join(sorted(required - columns)))
        if set(filters) - columns:
            raise ValueError(f"{path.name} is missing filter columns: " + ", ".join(sorted(set(filters) - columns)))
        offset = 0
        for batch in parquet.iter_batches(batch_size=512, columns=projection):
            for row in batch.to_pylist():
                position = offset
                offset += 1
                scanned += 1
                entry_id = row["entry_id"]
                match = _ENTRY.fullmatch(entry_id) if isinstance(entry_id, str) else None
                if not match:
                    exclusions["malformed_entry_id_rows"] += 1
                    continue
                pair_id, arm = match.groups()
                if isinstance(row[split_column], str):
                    available_splits.add(row[split_column])
                if entry_id in seen_entries:
                    invalid_pairs.add(pair_id)
                    exclusions["duplicate_entry_id_rows"] += 1
                    continue
                seen_entries.add(entry_id)
                expected = ("attack", "yes") if arm == "malicious" else ("control", "no")
                if (row["arm"], row["label"]) != expected:
                    invalid_pairs.add(pair_id)
                    exclusions["inconsistent_arm_label_rows"] += 1
                    continue
                index.setdefault(pair_id, {})[arm] = (file_index, position, row)
    if split is not None and split not in available_splits:
        if split == "test" and source is None:
            from .test_inputs import LEADERBOARD_SPACE
            raise ValueError(
                "The BrowseSafe++ test split is reserved for the leaderboard and its labels are not public. "
                "Load the label-free test pages with browsesafe_pp.load_test_inputs() (or run `browsesafe "
                f"predict`) and upload the predictions to the leaderboard Space ({LEADERBOARD_SPACE}). Load "
                "split=\"train\" from the Hub, or pass source= for a local copy that contains a test partition.")
        raise ValueError(f"Split {split!r} is absent; available splits: " + ", ".join(sorted(available_splits)))
    eligible = []
    exclusions["invalid_pairs"] = len(invalid_pairs)
    for pair_id, arms in index.items():
        if pair_id in invalid_pairs:
            continue
        if set(arms) != {"malicious", "benign"}:
            exclusions["incomplete_pairs"] += 1
            continue
        malicious, benign = arms["malicious"][2], arms["benign"][2]
        if malicious[split_column] != benign[split_column] or not isinstance(malicious[split_column], str):
            exclusions["inconsistent_split_pairs"] += 1
            continue
        if split is not None and malicious[split_column] != split:
            exclusions["other_split_pairs"] += 1
            continue
        if not _matches(malicious, filters):
            exclusions["metadata_filtered_pairs"] += 1
            continue
        eligible.append(pair_id)
    eligible.sort(key=lambda key: (hashlib.sha256(f"{identity}\0{key}\0{seed}".encode()).digest(), key))
    selected = eligible if n_pairs is None else eligible[:n_pairs]
    locations: dict[int, dict[int, tuple[str, str]]] = {}
    for pair_id in selected:
        for arm, (file_index, position, _) in index[pair_id].items():
            locations.setdefault(file_index, {})[position] = (pair_id, arm)
    del index
    manifest = {"schema_version": "1", **source_info, "task": task, "split": split,
                "split_column": split_column, "view": view, "seed": seed, "filters": filters,
                "requested_pairs": n_pairs, "scanned_rows": scanned,
                "eligible_pairs_before_sampling": len(eligible), "selected_pair_ids": selected,
                "sampling_identity": identity, "sampling_algorithm": "sha256(identity\\0pair_id\\0seed)"}
    return files, locations, manifest, exclusions


def screenshot_path(root: Path, entry_id: str) -> Path | None:
    """A page's capture: flat ``<root>/<entry_id>.jpg`` or bucketed ``<root>/<first two characters>/<entry_id>.jpg``."""
    for candidate in (root / f"{entry_id}.jpg", root / entry_id[:2] / f"{entry_id}.jpg"):
        if candidate.is_file():
            return candidate
    return None


def _screenshot_root(files: list[Path]) -> Path:
    """The ``screenshots/`` directory beside the ``data/`` directory of ``files``."""
    data = files[0].parent
    screenshots = (data.parent if data.name == "data" else data) / "screenshots"
    if not screenshots.is_dir():
        raise FileNotFoundError("This source has no screenshots/ directory")
    return screenshots


class _Screenshots:
    """Local JPEG paths for the selected pages.

    A ``screenshot`` cell with bytes (dataset v0.2 and later) is written once to
    ``<cache>/<entry_id>.jpg``; an existing file of the same size is reused. A page without
    embedded bytes uses the source's screenshot files, which ``files_root`` locates (and, for
    the Hub, downloads) the first time such a page is read.
    """

    def __init__(self, cache: Path, files_root):
        self.cache = cache
        self._files_root = files_root
        self._root: Path | None = None

    def path(self, entry_id: str, cell: Any) -> Path | None:
        data = cell.get("bytes") if isinstance(cell, Mapping) else None
        if data:
            if not _SAFE_NAME.fullmatch(entry_id):
                raise ValueError(f"Page id {entry_id!r} cannot name a cache file")
            target = self.cache / f"{entry_id}.jpg"
            if not (target.is_file() and target.stat().st_size == len(data)):
                self.cache.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
                partial.write_bytes(data)
                os.replace(partial, target)
            return target
        if self._root is None:
            self._root = self._files_root()
        return screenshot_path(self._root, entry_id)


def _screenshot_source(files: list[Path], info: Mapping[str, Any], source: str | Path | None,
                       token: str | bool | None, offline: bool, subdir: str = "") -> _Screenshots:
    """The screenshot cache under ``$HF_HOME/browsesafe_pp/screenshots/``, keyed by the dataset revision
    (local sources: by a hash of the source path), with the source's screenshot files as the fallback."""
    from huggingface_hub import constants

    base = Path(constants.HF_HOME) / "browsesafe_pp" / "screenshots"
    if source is not None:
        location = str(Path(source).expanduser().resolve())
        return _Screenshots(base / f"local-{hashlib.sha256(location.encode()).hexdigest()[:16]}",
                            lambda: _screenshot_root(files))
    repo_id = info.get("dataset") or dataset_id(None)
    revision = info.get("revision") or info.get("requested_revision") or DEFAULT_REVISION
    key = revision if _SHA.fullmatch(revision) else hashlib.sha256(revision.encode()).hexdigest()[:16]
    prefix = f"{subdir}/" if subdir else ""

    def download() -> Path:
        from huggingface_hub import snapshot_download

        root = Path(_hub_call(repo_id, snapshot_download, repo_id=repo_id, repo_type="dataset", revision=revision,
                              token=token, local_files_only=offline or bool(constants.HF_HUB_OFFLINE),
                              allow_patterns=[f"{prefix}screenshots/**"]))
        if not (root / prefix / "screenshots").is_dir():
            raise FileNotFoundError("This source has no screenshots/ directory")
        return root / prefix / "screenshots"

    return _Screenshots(base / repo_id.replace("/", "--") / key, download)


def _screenshot_columns(schema, view: str) -> list[str]:
    """The ``screenshot`` column to read: the whole cell for the screenshot view, else only its path."""
    import pyarrow as pa

    if "screenshot" not in schema.names:
        return []
    kind = schema.field("screenshot").type
    if view != SCREENSHOT_VIEW and pa.types.is_struct(kind) and kind.get_field_index("path") >= 0:
        return ["screenshot.path"]
    return ["screenshot"]


def _readable_image(path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("The screenshot view needs Pillow: pip install 'browsesafe-pp[screenshot] @ git+https://github.com/perplexityai/browsesafe-plus-plus.git'") from exc
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def _iter_pages(files: list[Path], locations: dict[int, dict[int, tuple[str, str]]],
                view: str, exclusions: Counter[str], screenshots: _Screenshots | None = None
                ) -> Iterator[tuple[str, str, Page]]:
    import pyarrow.parquet as pq

    content_column = VIEWS.get(view)
    for file_index in sorted(locations):
        positions = locations[file_index]
        with pq.ParquetFile(files[file_index]) as parquet:
            columns = [name for name in parquet.schema_arrow.names if name != "screenshot"
                       and (not name.startswith("content_") or name == content_column)]
            columns += _screenshot_columns(parquet.schema_arrow, view)
            offset = 0
            for group_index in range(parquet.num_row_groups):
                group_size = parquet.metadata.row_group(group_index).num_rows
                if not any(offset <= position < offset + group_size for position in positions):
                    offset += group_size
                    continue
                for batch in parquet.iter_batches(batch_size=8, row_groups=[group_index], columns=columns):
                    for i in range(batch.num_rows):
                        if offset + i not in positions:
                            continue
                        row = batch.slice(i, 1).to_pylist()[0]
                        pair_id, arm = positions[offset + i]
                        cell = row.get("screenshot")
                        if isinstance(cell, dict) and "bytes" in cell:
                            row["screenshot"] = {"path": cell.get("path")}  # metadata keeps the reference only
                        if view == SCREENSHOT_VIEW:
                            shot = screenshots.path(row["entry_id"], cell)
                            if shot is None or not _readable_image(shot):
                                exclusions["missing_view_members"] += 1
                                continue
                            content = str(shot)
                            evidence, counts = _references(row, "")
                        else:
                            content = row.pop(content_column)
                            if not isinstance(content, str) or not content.strip():
                                exclusions["missing_view_members"] += 1
                                continue
                            evidence, counts = _references(row, content)
                        row["evidence_binding"] = counts
                        yield pair_id, arm, Page(entry_id=row["entry_id"], content=content, view=view,
                                                 task=row.get("benign_task"), injected=arm == "malicious",
                                                 metadata=row, evidence=evidence)
                    offset += batch.num_rows


def load(*, task: str = "detection", split: str | None = DEFAULT_SPLIT, view: str = "html",
         source: str | Path | None = None, dataset: str | None = None,
         revision: str = DEFAULT_REVISION, token: str | bool | None = None,
         n_pairs: int | None = None, seed: int = 42, filters: Mapping[str, Any] | None = None,
         offline: bool = False, split_column: str = "split") -> Benchmark:
    """Load complete attack/control pairs.

    Args:
        task: ``"detection"`` (default) or ``"evidence"`` (inspect injection references).
        split: ``"train"`` (default), ``"test"`` or ``None`` for every partition. The Hub
            dataset's labelled pairs are ``"train"``; the test split is reserved for the leaderboard
            (its label-free pages load with :func:`~browsesafe_pp.load_test_inputs`).
        view: ``"html"``, ``"markdown"``, ``"accessibility"`` or ``"screenshot"``.
        source: a local directory of ``members-*.parquet`` shards (or its parent containing ``data/``).
            When omitted, the shards are downloaded from the gated Hugging Face dataset: request
            access on its page and log in first.
        dataset: Hugging Face dataset id; defaults to ``$BROWSESAFE_DATASET``, else :data:`DEFAULT_DATASET`.
        revision: branch, tag or commit of the dataset. Pin a 40-character commit for comparable runs.
        token: Hugging Face access token; by default the token of ``hf auth login`` or ``$HF_TOKEN``.
        n_pairs: maximum number of pairs; ``None`` loads every eligible pair.
        seed: sampling seed. The same source, seed and arguments select the same pairs.
        filters: constraints on the malicious member's metadata. List-valued fields (languages)
            match when any requested value is present; other fields use exact membership.
        offline: use an already downloaded snapshot without contacting the Hub.
        split_column: column that holds the partition name.

    Pages whose selected view is missing are excluded together with their mate and
    counted in ``manifest["exclusions"]``; no replacement pairs are sampled.
    """
    files, locations, manifest, exclusions = _prepare_load(
        task=task, split=split, split_column=split_column, view=view, source=source, dataset=dataset,
        revision=revision, token=token, n_pairs=n_pairs, seed=seed, filters=filters, offline=offline)
    screenshots = _screenshot_source(files, manifest, source, token, offline) if view == SCREENSHOT_VIEW else None
    loaded: dict[str, dict[str, Page]] = {}
    for pair_id, arm, page in _iter_pages(files, locations, view, exclusions, screenshots):
        loaded.setdefault(pair_id, {})[arm] = page
    pairs = []
    for pair_id in manifest["selected_pair_ids"]:
        arms = loaded.get(pair_id, {})
        if set(arms) != {"malicious", "benign"}:
            exclusions["missing_view_pairs"] += 1
            continue
        pairs.append(Pair(pair_id, arms["malicious"], arms["benign"]))
    return _loaded_benchmark(pairs, manifest, exclusions)


def _loaded_benchmark(pairs: list[Pair], manifest: dict[str, Any], exclusions: Counter[str]) -> Benchmark:
    evidence_counts: Counter[str] = Counter()
    for pair in pairs:
        evidence_counts.update(pair.malicious.metadata["evidence_binding"])
        evidence_counts["malicious_members_with_bound_references"] += bool(pair.malicious.evidence)
    pair_ids = [pair.pair_id for pair in pairs]
    requested = manifest["requested_pairs"]
    return Benchmark(tuple(pairs), {
        **manifest, "pair_ids": pair_ids, "loaded_pairs": len(pairs), "loaded_pages": 2 * len(pairs),
        "selection_shortfall": max(0, requested - len(pairs)) if requested is not None else 0,
        "selection_digest": hashlib.sha256(json.dumps(pair_ids).encode()).hexdigest(),
        "exclusions": dict(exclusions), "evidence_coverage": dict(evidence_counts),
    })


def iter_batches(*, batch_pairs: int = 64, order: str = "selection",
                 max_spool_bytes: int = 10 * 1024**3, spool_dir: str | Path | None = None,
                 task: str = "detection", split: str | None = DEFAULT_SPLIT, view: str = "html",
                 source: str | Path | None = None, dataset: str | None = None,
                 revision: str = DEFAULT_REVISION, token: str | bool | None = None,
                 n_pairs: int | None = None, seed: int = 42, filters: Mapping[str, Any] | None = None,
                 offline: bool = False, split_column: str = "split") -> Iterator[Benchmark]:
    """Yield the same pairs as :func:`load` in bounded batches.

    ``order="selection"`` keeps :func:`load`'s order by spooling the selected pages to a
    temporary SQLite file first; ``order="source"`` emits pairs as soon as both members
    have been read, which spools far less. The spool lives in ``spool_dir`` (default: the
    system temporary directory), is capped at ``max_spool_bytes`` and is deleted when the
    iterator finishes or is closed. The last batch has ``manifest["batch"]["is_final"]``
    set and carries the exclusions for the whole selection.
    """
    if type(batch_pairs) is not int or batch_pairs < 1:
        raise ValueError("batch_pairs must be a positive integer")
    if order not in {"selection", "source"}:
        raise ValueError("order must be 'selection' or 'source'")
    if type(max_spool_bytes) is not int or max_spool_bytes < 16384:
        raise ValueError("max_spool_bytes must be an integer of at least 16384")
    files, locations, manifest, exclusions = _prepare_load(
        task=task, split=split, split_column=split_column, view=view, source=source, dataset=dataset,
        revision=revision, token=token, n_pairs=n_pairs, seed=seed, filters=filters, offline=offline)
    screenshots = _screenshot_source(files, manifest, source, token, offline) if view == SCREENSHOT_VIEW else None
    temporary = tempfile.TemporaryDirectory(prefix="browsesafe-pp-", dir=spool_dir)
    try:
        connection = sqlite3.connect(Path(temporary.name) / "pages.sqlite3")
        try:
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA journal_mode=OFF")
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count={max_spool_bytes // page_size}")
            connection.execute("CREATE TABLE pages (pair_id TEXT, arm TEXT, page TEXT, PRIMARY KEY (pair_id, arm))")
            available: Counter[str] = Counter()
            total_loaded = 0

            def read_pair(pair_id: str) -> Pair:
                arms = {}
                for arm, raw in connection.execute("SELECT arm, page FROM pages WHERE pair_id=?", (pair_id,)):
                    fields = json.loads(raw)
                    fields["evidence"] = tuple(fields["evidence"])
                    arms[arm] = Page(**fields)
                return Pair(pair_id, arms["malicious"], arms["benign"])

            def pair_batches() -> Iterator[list[Pair]]:
                nonlocal total_loaded
                pairs: list[Pair] = []
                for pair_id, arm, page in _iter_pages(files, locations, view, exclusions, screenshots):
                    raw = json.dumps(asdict(page), ensure_ascii=False)
                    if shutil.disk_usage(temporary.name).free < 2 * len(raw.encode()) + 65536:
                        raise OSError("Insufficient free disk space for the page spool")
                    connection.execute("INSERT INTO pages VALUES (?, ?, ?)", (pair_id, arm, raw))
                    available[pair_id] += 1
                    if order == "source" and available[pair_id] == 2:
                        pairs.append(read_pair(pair_id))
                        connection.execute("DELETE FROM pages WHERE pair_id=?", (pair_id,))
                        total_loaded += 1
                        if len(pairs) == batch_pairs:
                            yield pairs
                            pairs = []
                connection.commit()
                exclusions["missing_view_pairs"] += sum(
                    available[pair_id] != 2 for pair_id in manifest["selected_pair_ids"])
                if order == "selection":
                    for pair_id in manifest["selected_pair_ids"]:
                        if available[pair_id] != 2:
                            continue
                        pairs.append(read_pair(pair_id))
                        total_loaded += 1
                        if len(pairs) == batch_pairs:
                            yield pairs
                            pairs = []
                if pairs:
                    yield pairs

            def wrap(pairs: list[Pair], batch_index: int, is_final: bool) -> Benchmark:
                batch = _loaded_benchmark(pairs, manifest, exclusions)
                batch.manifest["selection_shortfall"] = (
                    max(0, n_pairs - total_loaded) if n_pairs is not None else 0) if is_final else None
                batch.manifest["batch"] = {
                    "batch_index": batch_index, "batch_pairs": batch_pairs, "is_final": is_final,
                    "batch_count": batch_index + 1 if is_final else None,
                    "total_loaded_pairs": total_loaded if is_final else None,
                    "total_selected_pairs": len(manifest["selected_pair_ids"]), "ordering": order,
                }
                return batch

            pending: list[Pair] | None = None
            batch_index = 0
            for pairs in pair_batches():
                if pending is not None:
                    yield wrap(pending, batch_index, False)
                    batch_index += 1
                pending = pairs
            yield wrap(pending or [], batch_index, True)
        finally:
            connection.close()
    finally:
        temporary.cleanup()


def describe(*, source: str | Path | None = None, dataset: str | None = None,
             revision: str = DEFAULT_REVISION, token: str | bool | None = None,
             offline: bool = False, split_column: str = "split") -> dict[str, Any]:
    """Count pages, pairs, splits and page types from the metadata columns only."""
    import pyarrow.parquet as pq

    files, source_info, _ = _files(source, dataset, revision, token, offline)
    counts: dict[str, Counter[str]] = {key: Counter() for key in ("split", "page_type", "label")}
    views: Counter[str] = Counter()
    pages = 0
    pair_ids: set[str] = set()
    for path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        for view, column in VIEWS.items():
            views[view] += column in names
        columns = sorted({"entry_id", "label", split_column, "page_type"} & names)
        for batch in parquet.iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                pages += 1
                match = _ENTRY.fullmatch(str(row.get("entry_id")))
                if match:
                    pair_ids.add(match.group(1))
                malicious = bool(match and match.group(2) == "malicious")
                counts["label"][str(row.get("label"))] += 1
                if malicious:
                    counts["split"][str(row.get(split_column))] += 1
                    counts["page_type"][str(row.get("page_type"))] += 1
    return {**{k: v for k, v in source_info.items() if k != "files"}, "pages": pages, "pairs": len(pair_ids),
            "pairs_by_split": dict(counts["split"]),
            "pairs_by_page_type": dict(counts["page_type"].most_common()),
            "pages_by_label": dict(counts["label"]),
            "text_views": [view for view in VIEWS if views[view] == len(files)]}


def __getattr__(name: str) -> Any:
    # Back-compat: detection scoring moved to browsesafe_pp.evaluation.classification.
    if name in {"DetectionReport", "evaluate", "evaluate_async"}:
        from .evaluation import classification
        return getattr(classification, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ALL_VIEWS", "Benchmark", "DATASET_ENV", "DEFAULT_DATASET", "DEFAULT_REVISION", "DetectorInput",
           "FILTERS", "Page", "Pair", "Prediction", "SCREENSHOT_VIEW", "VIEWS", "dataset_id", "describe",
           "iter_batches", "load", "screenshot_path"]
