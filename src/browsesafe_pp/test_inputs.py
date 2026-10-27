"""Label-free test pages and prediction files for the BrowseSafe++ leaderboard.

The test split of the gated BrowseSafe++ dataset (files under ``test/``) is published
without labels: one row per page in the training split's columns, with an opaque
``entry_id``, the benign task and the four views, and every label and annotation empty.
Run a detector on it locally, write the decisions as
JSON Lines and upload the file to the leaderboard Space, which scores it against the
private labels with the same metrics as :func:`browsesafe_pp.evaluate`. See
``docs/leaderboard.md``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import dataset as _data
from .dataset import ALL_VIEWS, DEFAULT_DATASET, DEFAULT_REVISION, SCREENSHOT_VIEW, VIEWS, DetectorInput
from .evaluation.classification import Outcome, _outcomes, _outcomes_async

TEST_DATASET = DEFAULT_DATASET  # one repository holds the training split and the test inputs
LEADERBOARD_SPACE = "https://huggingface.co/spaces/perplexity-ai/browsesafe-plus-plus-leaderboard"
TEST_DIRECTORY = "test"  # test/data/test-*.parquet, test/screenshots/<id[:2]>/<id>.jpg
TEST_SHARD_PATTERN = "test-*.parquet"
_RECORD_KEYS = ("id", "view", "injected", "score")


@dataclass(frozen=True)
class TestPage:
    """One test page in the selected view; ``id`` is the row's opaque ``entry_id``. It carries no label, arm or pair."""

    __test__ = False  # not a pytest test class

    id: str
    content: str
    view: str
    task: str | None

    @property
    def model_input(self) -> DetectorInput:
        """The default, task-free detector input (as :attr:`browsesafe_pp.Page.model_input`)."""
        return DetectorInput(content=self.content, view=self.view)


def load_test_inputs(*, view: str = "html", source: str | Path | None = None, dataset: str | None = None,
                     revision: str = DEFAULT_REVISION, token: str | bool | None = None, offline: bool = False,
                     limit: int | None = None) -> tuple[TestPage, ...]:
    """Load the leaderboard test pages in one view, in dataset order.

    Args:
        view: ``"html"``, ``"markdown"``, ``"accessibility"`` or ``"screenshot"`` (``content`` is the
            local JPEG path).
        source: a local copy of the dataset with ``test/data/test-*.parquet`` (and, for screenshots,
            ``test/screenshots/``), or its ``test/`` directory. When omitted, only the test files are
            downloaded from the gated Hub dataset; request access and log in first.
        dataset: Hugging Face dataset id; defaults to ``$BROWSESAFE_DATASET``, else :data:`DEFAULT_DATASET`
            (the same repository as the training split).
        revision, token, offline: as in :func:`browsesafe_pp.load`.
        limit: only the first ``limit`` pages, for a trial run. A submission needs every page.

    A page without the requested view raises an error instead of being dropped, because a
    missing page counts as an error in the leaderboard's scores.
    """
    import pyarrow.parquet as pq

    if view not in ALL_VIEWS:
        raise ValueError(f"view must be one of {ALL_VIEWS}")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    files, info, _ = _data._files(source, dataset, revision, token, offline, subdir=TEST_DIRECTORY,
                                  pattern=TEST_SHARD_PATTERN)
    column = VIEWS.get(view)
    screenshots = (_data._screenshot_source(files, info, source, token, offline, TEST_DIRECTORY)
                   if view == SCREENSHOT_VIEW else None)
    pages: list[TestPage] = []
    seen: set[str] = set()
    for path in files:
        with pq.ParquetFile(path) as parquet:
            names = set(parquet.schema_arrow.names)
            # The page id is ``entry_id`` (the training split's schema); older layouts used ``id``.
            key = "entry_id" if "entry_id" in names or "id" not in names else "id"
            required = {key} | ({column} if column else set())
            if required - names:
                raise ValueError(f"{path.name} is missing required columns: " + ", ".join(sorted(required - names)))
            columns = sorted(required | ({"benign_task"} & names))
            if screenshots is not None and "screenshot" in names:
                columns.append("screenshot")
            for batch in parquet.iter_batches(batch_size=8 if screenshots else 64, columns=columns):
                for row in batch.to_pylist():
                    if limit is not None and len(pages) >= limit:
                        return tuple(pages)
                    page_id = row[key]
                    if not isinstance(page_id, str) or not page_id or page_id in seen:
                        raise ValueError(f"Missing or duplicate test page id: {page_id!r}")
                    seen.add(page_id)
                    if screenshots is not None:
                        shot = screenshots.path(page_id, row.get("screenshot"))
                        if shot is None:
                            raise FileNotFoundError(f"No screenshot for test page {page_id}")
                        content = str(shot)
                    else:
                        content = row[column]
                        if not isinstance(content, str) or not content.strip():
                            raise ValueError(f"Test page {page_id} has no {view} content")
                    pages.append(TestPage(page_id, content, view, row.get("benign_task")))
    return tuple(pages)


def _records(pages: Sequence[TestPage], outcomes: list[Outcome]) -> list[dict[str, Any]]:
    records = []
    for page, (status, injected, value) in zip(pages, outcomes):
        if status not in {"valid", "abstained"}:
            continue  # errors and invalid predictions are left out; the Space counts them as errors
        record: dict[str, Any] = {"id": page.id, "view": page.view, "injected": injected}
        if value is not None:
            record["score"] = float(value)
        records.append(record)
    return records


def predict_test(pages: Sequence[TestPage], predict, *, concurrency: int = 1) -> list[dict[str, Any]]:
    """Call ``predict`` once per page (no retries) and return records in the leaderboard format.

    ``predict`` follows the detector contract of :func:`browsesafe_pp.evaluate` (sync or async;
    ``concurrency`` bounds the async calls in flight). Each record is ``{"id", "view", "injected"}``
    plus ``"score"`` when the detector gave one; an abstention writes ``"injected": None``.
    Exceptions and invalid return values leave the page out of the records.
    """
    if not callable(predict):
        raise TypeError("predict must be callable")
    if not (inspect.iscoroutinefunction(predict) or inspect.iscoroutinefunction(getattr(predict, "__call__", None))):
        return _records(pages, _outcomes(pages, predict))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(predict_test_async(pages, predict, concurrency=concurrency))
    raise RuntimeError("An event loop is running; use `await predict_test_async(...)` for an async detector")


async def predict_test_async(pages: Sequence[TestPage], predict, *, concurrency: int = 8) -> list[dict[str, Any]]:
    """Like :func:`predict_test`, awaited inside a running event loop; records keep page order."""
    if not callable(predict):
        raise TypeError("predict must be callable")
    return _records(pages, await _outcomes_async(pages, predict, concurrency))


def _clean(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) - set(_RECORD_KEYS) or "injected" not in record:
        raise ValueError(f"A prediction record needs id, view and injected (and optionally score): {record!r}")
    page_id, view, injected, value = (record.get(key) for key in _RECORD_KEYS)
    if not isinstance(page_id, str) or not page_id:
        raise ValueError(f"Invalid id in prediction record: {record!r}")
    if view not in ALL_VIEWS:
        raise ValueError(f"view must be one of {ALL_VIEWS}: {record!r}")
    if injected is not None and type(injected) is not bool:
        raise ValueError(f"injected must be true, false or null: {record!r}")
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"score must be a number in [0, 1]: {record!r}")
    clean = {"id": page_id, "view": view, "injected": injected}
    if value is not None:
        clean["score"] = float(value)
    return clean


def write_predictions(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Validate ``records`` and write them as JSON Lines to the new file ``path``.

    The file is what the leaderboard Space accepts: one record per (``id``, ``view``). It is
    written only when every record is valid and no (``id``, ``view``) repeats.
    """
    lines = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        clean = _clean(record)
        key = (clean["id"], clean["view"])
        if key in seen:
            raise ValueError(f"duplicate record for id {key[0]!r} and view {key[1]!r}")
        seen.add(key)
        lines.append(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.writelines(lines)
    return path


__all__ = ["LEADERBOARD_SPACE", "TEST_DATASET", "TestPage", "load_test_inputs", "predict_test",
           "predict_test_async", "write_predictions"]
