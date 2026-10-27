import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import browsesafe_pp as bs
from browsesafe_pp import cli, testing
from browsesafe_pp.testing import rows, write_fixture, write_test_fixture

DATASET_URL = "https://huggingface.co/datasets/perplexity-ai/browsesafe-plus-plus"
# The published columns, in order; the train and test splits share them (one configuration).
COLUMNS = ["entry_id", "screenshot", "label", "arm", "title", "split", "injection_text_fragments_json", "benign_task",
           "attack_identified_ts", "attack_quotes_json", "attack_type", "expected_outcome", "goal_class",
           "injection_strategy", "malicious_goal", "register", "risk_category", "language_page",
           "requires_action_tools", "linguistic_style", "language_payload", "page_type", "domain", "content_markdown",
           "content_ax_yaml", "content_html"]
FILLED = {"entry_id", "split", "benign_task", "screenshot", "content_markdown", "content_ax_yaml", "content_html"}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    return write_test_fixture(tmp_path_factory.mktemp("test-inputs"), n_pairs=6)


def labels(root):
    return {row["id"]: row for row in pq.read_table(root / "labels.parquet").to_pylist()}


def test_load_test_inputs_fixture(inputs):
    assert not (inputs / "data").exists() and not (inputs / "screenshots").exists()
    table = pq.read_table(inputs / "test" / "data" / "test-0000.parquet")
    assert table.column_names == COLUMNS
    ids = table.column("entry_id").to_pylist()
    truth = labels(inputs)
    assert len(ids) == len(set(ids)) == 6 and set(ids) == set(truth)  # one page per pair
    assert sorted(row["arm"] for row in truth.values()) == ["benign"] * 3 + ["malicious"] * 3
    assert len({row["pair_id"] for row in truth.values()}) == 6
    for view, marker in (("html", "<main>"), ("markdown", "# "), ("accessibility", "- main:")):
        pages = bs.load_test_inputs(source=inputs, view=view)
        assert [page.id for page in pages] == ids
        assert all(page.view == view and marker in page.content for page in pages)
        assert pages[0].task == "What are the opening hours?"
        assert pages[0].model_input == bs.DetectorInput(content=pages[0].content, view=view)
        assert not hasattr(pages[0], "injected")
    shots = bs.load_test_inputs(source=inputs, view="screenshot")
    root = inputs.resolve() / "test" / "screenshots"
    assert table.column("screenshot").to_pylist()[0]["path"].startswith(
        f"hf://datasets/{bs.DEFAULT_DATASET}@main/test/screenshots/")
    assert [Path(page.content).name for page in shots] == [f"{i}.jpg" for i in ids]  # embedded, cached locally
    assert all(Path(page.content).read_bytes() == (root / page.id[:2] / f"{page.id}.jpg").read_bytes()
               for page in shots)
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    assert sorted(entry["entry_id"] for entry in manifest) == sorted(ids)
    assert [page.id for page in bs.load_test_inputs(source=inputs, limit=5)] == ids[:5]
    with pytest.raises(ValueError):
        bs.load_test_inputs(source=inputs, view="pdf")


def test_test_shard_has_the_training_schema(tmp_path):
    import re

    root = write_test_fixture(write_fixture(tmp_path / "both", n_pairs=12), n_pairs=6)
    train = pq.read_schema(root / "data" / "members-00000.parquet")
    test = pq.read_schema(root / "test" / "data" / "test-0000.parquet")
    assert train.names == COLUMNS and test.equals(train, check_metadata=True)
    features = json.loads(train.metadata[b"huggingface"])["info"]["features"]
    assert list(features) == COLUMNS and features["screenshot"] == {"_type": "Image"}
    rows_ = pq.read_table(root / "test" / "data" / "test-0000.parquet").to_pylist()
    assert len(rows_) == 6
    for row in rows_:
        page_id = row["entry_id"]
        assert re.fullmatch(r"[0-9a-f]{12}", page_id) and row["split"] == "test"
        assert row["screenshot"] == {"bytes": testing._JPEG, "path": f"hf://datasets/{bs.DEFAULT_DATASET}@main/"
                                                                    f"test/screenshots/{page_id[:2]}/{page_id}.jpg"}
        assert all(isinstance(row[key], str) and row[key] for key in FILLED - {"screenshot"})
        assert {key for key, value in row.items() if value is not None} == FILLED  # labels and annotations null
    truth = pq.read_table(root / "labels.parquet")
    assert truth.column_names == ["id", "entry_id", "pair_id", "arm", "injected", "test_version", "html_sha256"]
    assert sorted(truth.column("id").to_pylist()) == sorted(row["entry_id"] for row in rows_)
    assert all(re.fullmatch(r"\d{5}_(malicious|benign)", entry) for entry in truth.column("entry_id").to_pylist())
    shots = pq.read_table(root / "data" / "members-00000.parquet").column("screenshot").to_pylist()
    assert shots[0]["path"] == f"hf://datasets/{bs.DEFAULT_DATASET}@main/screenshots/00/00001_malicious.jpg"


def test_older_layout_with_an_id_column_still_loads(tmp_path):
    (tmp_path / "test" / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": "0123456789ab", "benign_task": "Summarize.", "content_html": "<p>a</p>"},
                                         {"id": "ba9876543210", "benign_task": None, "content_html": "<p>b</p>"}]),
                   tmp_path / "test" / "data" / "test-0000.parquet")
    pages = bs.load_test_inputs(source=tmp_path)
    assert [(page.id, page.task) for page in pages] == [("0123456789ab", "Summarize."), ("ba9876543210", None)]


def train_screenshots(root, n_pairs, split=None):
    for row in rows(n_pairs):
        if split is None or row["split"] == split:
            path = root / "screenshots" / row["entry_id"][:2] / f"{row['entry_id']}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(testing._JPEG)
    return root


class FakeHub:
    """Serve a local directory as a Hub dataset: each download materializes only the requested files."""

    def __init__(self, monkeypatch, repo, cache, *, info_error=None, download_error=None):
        import huggingface_hub
        from huggingface_hub.utils import filter_repo_objects

        self.calls, self.selected = [], []
        files = sorted(path.relative_to(repo).as_posix() for path in repo.rglob("*") if path.is_file())
        hub = self

        class Api:
            def __init__(self, token=None):
                pass

            def dataset_info(self, repo_id, revision=None):
                hub.calls.append({"call": "dataset_info", "repo_id": repo_id, "revision": revision})
                if info_error is not None:
                    raise info_error
                return SimpleNamespace(sha="c" * 40)

        def snapshot_download(**kwargs):
            hub.calls.append({"call": "snapshot_download", **kwargs})
            if download_error is not None:
                raise download_error
            selected = sorted(filter_repo_objects(files, allow_patterns=kwargs["allow_patterns"]))
            hub.selected.append(selected)
            root = cache / str(len(hub.selected)) / kwargs["revision"]
            for name in selected:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / name, root / name)
            root.mkdir(parents=True, exist_ok=True)
            return str(root)

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)


@pytest.fixture
def hub_repo(tmp_path):
    """The published layout: labelled train shards and screenshots, label-free test inputs under test/."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row for row in rows(12) if row["split"] == "train"]),
                   repo / "data" / "members-train-00000.parquet")
    train_screenshots(repo, 12, split="train")
    write_test_fixture(repo, n_pairs=6, embed_screenshots=False)  # a v0.1 revision: no images in the shards
    (repo / "labels.parquet").unlink()  # the labels are never in the public repository
    return repo


def test_embedded_screenshots_need_no_screenshot_files(tmp_path, monkeypatch, hf_home):
    import os

    repo = write_test_fixture(train_screenshots(write_fixture(tmp_path / "v02", n_pairs=12), 12), n_pairs=6)
    (repo / "labels.parquet").unlink()
    hub = FakeHub(monkeypatch, repo, tmp_path / "cache")
    monkeypatch.delenv("BROWSESAFE_DATASET", raising=False)
    cache = hf_home / "browsesafe_pp" / "screenshots" / "perplexity-ai--browsesafe-plus-plus" / ("c" * 40)

    shots = bs.load(view="screenshot")
    assert shots.manifest["loaded_pairs"] == 6 and len(hub.selected) == 1
    assert hub.selected[0] == ["data/members-00000.parquet", "data/members-00001.parquet"]  # no screenshots/**
    assert sorted(Path(page.content) for page in shots) == sorted(cache / f"{page.entry_id}.jpg" for page in shots)
    assert all(Path(page.content).read_bytes() == testing._JPEG for page in shots)
    assert len(list(cache.iterdir())) == 12  # only the selected pages are written
    assert shots.pairs[0].malicious.metadata["screenshot"] == {
        "path": f"hf://datasets/{bs.DEFAULT_DATASET}@main/screenshots/{shots.pairs[0].pair_id[:2]}/"
                f"{shots.pairs[0].malicious.entry_id}.jpg"}  # the reference stays; the bytes do not

    test_shots = bs.load_test_inputs(view="screenshot", limit=3)
    assert hub.selected[-1] == ["test/data/test-0000.parquet"] and len(hub.selected) == 2
    assert [Path(page.content) for page in test_shots] == [cache / f"{page.id}.jpg" for page in test_shots]
    assert len(list(cache.iterdir())) == 15

    kept, damaged = Path(shots.pairs[0].malicious.content), Path(shots.pairs[0].benign.content)
    os.utime(kept, ns=(1, 1))
    damaged.write_bytes(b"short")
    again = bs.load(view="screenshot")
    assert again.manifest["loaded_pairs"] == 6
    assert kept.stat().st_mtime_ns == 1  # same size: reused
    assert damaged.read_bytes() == testing._JPEG  # different size: rewritten
    batches = list(bs.iter_batches(view="screenshot", batch_pairs=4, spool_dir=tmp_path))
    assert sum(len(batch.pairs) for batch in batches) == 6
    assert not any("screenshots" in pattern for call in hub.calls if call["call"] == "snapshot_download"
                   for pattern in call["allow_patterns"])


def test_one_hub_dataset_holds_both_splits(hub_repo, tmp_path, monkeypatch):
    hub = FakeHub(monkeypatch, hub_repo, tmp_path / "cache")
    monkeypatch.delenv("BROWSESAFE_DATASET", raising=False)
    assert bs.TEST_DATASET == bs.DEFAULT_DATASET == "perplexity-ai/browsesafe-plus-plus"

    pages = bs.load_test_inputs(view="html")
    assert len(pages) == 6 and all(len(page.id) == 12 for page in pages)
    assert hub.calls[-1]["repo_id"] == bs.DEFAULT_DATASET
    assert hub.selected[-1] == ["test/data/test-0000.parquet"]

    shots = bs.load_test_inputs(view="screenshot")
    assert len(shots) == 6 and all("/test/screenshots/" in page.content for page in shots)
    assert all(name.startswith("test/") for name in hub.selected[-1])
    assert sum(name.endswith(".jpg") for name in hub.selected[-1]) == 6
    assert "test/screenshots/manifest.jsonl" in hub.selected[-1]

    train = bs.load()
    assert train.manifest["loaded_pairs"] == 6 and train.manifest["revision"] == "c" * 40
    assert hub.selected[-1] == ["data/members-train-00000.parquet"]
    with pytest.raises(ValueError) as error:
        bs.load(split="test")
    assert "load_test_inputs" in str(error.value) and bs.LEADERBOARD_SPACE in str(error.value)
    assert hub.selected[-1] == ["data/members-train-00000.parquet"]

    train_shots = bs.load(view="screenshot")
    assert train_shots.manifest["loaded_pairs"] == 6
    assert not any(name.startswith("test/") for name in hub.selected[-1])
    assert sum(name.endswith(".jpg") for name in hub.selected[-1]) == 12

    monkeypatch.setenv("BROWSESAFE_DATASET", "someone/mirror")
    bs.load_test_inputs()
    assert hub.calls[-1]["repo_id"] == "someone/mirror"
    bs.load_test_inputs(dataset="example/explicit")
    assert hub.calls[-1]["repo_id"] == "example/explicit"


@pytest.mark.parametrize("embedded", [True, False])
def test_combined_fixture_loads_through_both_loaders(tmp_path, hf_home, embedded):
    root = write_test_fixture(train_screenshots(write_fixture(tmp_path / "both", n_pairs=12,
                                                              embed_screenshots=embedded), 12),
                              n_pairs=6, embed_screenshots=embedded)
    cache = hf_home / "browsesafe_pp" / "screenshots"
    train = bs.load(source=root, split=None)
    assert train.manifest["loaded_pairs"] == 12
    assert [entry["name"] for entry in train.manifest["files"]] == ["members-00000.parquet", "members-00001.parquet"]
    assert bs.describe(source=root)["pages"] == 24
    train_shots = bs.load(source=root, split=None, view="screenshot")
    assert train_shots.manifest["loaded_pairs"] == 12
    if embedded:  # a local source is keyed by a hash of its path
        assert len({Path(page.content).parent for page in train_shots}) == 1
        assert all(Path(page.content).parents[1] == cache for page in train_shots)
    else:
        assert all(Path(page.content).parents[1] == root.resolve() / "screenshots" for page in train_shots)

    ids = pq.read_table(root / "test" / "data" / "test-0000.parquet").column("entry_id").to_pylist()
    for source in (root, root / "test", root / "test" / "data"):
        assert [page.id for page in bs.load_test_inputs(source=source)] == ids
    shots = bs.load_test_inputs(source=root, view="screenshot")
    if embedded:
        assert all(Path(page.content).parents[1] == cache and Path(page.content).name == f"{page.id}.jpg"
                   for page in shots)
        assert all(Path(page.content).read_bytes() == testing._JPEG for page in shots)
    else:
        assert all(Path(page.content).parents[1] == root.resolve() / "test" / "screenshots" for page in shots)

    with pytest.raises(FileNotFoundError):
        bs.load_test_inputs(source=write_fixture(tmp_path / "train-only", n_pairs=2))
    with pytest.raises(FileNotFoundError):
        bs.load(source=write_test_fixture(tmp_path / "test-only", n_pairs=2), split=None)


def http_error(cls, status):
    request = httpx.Request("GET", f"{DATASET_URL.replace('/datasets/', '/api/datasets/')}/revision/main")
    return cls(f"{status} Client Error.", response=httpx.Response(status, request=request))


def test_refused_hub_access_says_how_to_get_it(hub_repo, tmp_path, monkeypatch):
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    monkeypatch.delenv("BROWSESAFE_DATASET", raising=False)
    (tmp_path / "leaderboard_detector.py").write_text("def predict(request):\n    return False\n")
    monkeypatch.chdir(tmp_path)
    loaders = {"train": lambda: bs.load(), "test": lambda: bs.load_test_inputs(),
               "info": lambda: bs.describe(),
               "predict": lambda: cli.main(["predict", "--predict", "leaderboard_detector:predict", "--view",
                                            "html", "--output", str(tmp_path / "never.jsonl")])}
    refusals = [{"download_error": http_error(GatedRepoError, 403)},
                {"download_error": http_error(HfHubHTTPError, 403)},
                {"info_error": http_error(RepositoryNotFoundError, 401)},
                {"info_error": http_error(GatedRepoError, 401)}]
    for number, refusal in enumerate(refusals):
        for name, loader in loaders.items():
            hub = FakeHub(monkeypatch, hub_repo, tmp_path / f"cache-{number}-{name}", **refusal)
            with pytest.raises(PermissionError) as error:
                loader()
            message = str(error.value)
            assert DATASET_URL in message and "hf auth login" in message and "HF_TOKEN" in message, message
            failing = "snapshot_download" if "download_error" in refusal else "dataset_info"
            assert [call["call"] for call in hub.calls].count(failing) == 1  # no retries
    assert not (tmp_path / "never.jsonl").exists()

    FakeHub(monkeypatch, hub_repo, tmp_path / "cache-404", info_error=http_error(RepositoryNotFoundError, 404))
    with pytest.raises(RepositoryNotFoundError):
        bs.load_test_inputs()


def test_predict_test_records_format(inputs, tmp_path):
    pages = bs.load_test_inputs(source=inputs, view="markdown")
    behaviour = iter(["yes", "no", "abstain", "raise", "garbage", "score"] * 2)

    def detector(request):
        assert request.task is None and request.view == "markdown"
        kind = next(behaviour)
        if kind == "raise":
            raise RuntimeError("boom")
        if kind == "garbage":
            return "maybe"
        if kind == "abstain":
            return None
        if kind == "score":
            return bs.Prediction(True, score=1)
        return kind == "yes"

    records = bs.predict_test(pages, detector)
    assert len(records) == 4
    assert records[:4] == [{"id": pages[0].id, "view": "markdown", "injected": True},
                           {"id": pages[1].id, "view": "markdown", "injected": False},
                           {"id": pages[2].id, "view": "markdown", "injected": None},
                           {"id": pages[5].id, "view": "markdown", "injected": True, "score": 1.0}]
    assert type(records[3]["score"]) is float
    assert {r["id"] for r in records}.isdisjoint({pages[i].id for i in (3, 4)})

    async def slow(request):
        await asyncio.sleep(0.001 * (len(request.content) % 3))
        return "Assistant note" in request.content

    records = bs.predict_test(pages, slow, concurrency=4)
    truth = labels(inputs)
    assert [r["id"] for r in records] == [page.id for page in pages]
    assert all(r["injected"] == truth[r["id"]]["injected"] for r in records)

    async def inside_loop():
        with pytest.raises(RuntimeError, match="predict_test_async"):
            bs.predict_test(pages, slow)
        return await bs.predict_test_async(pages, slow, concurrency=4)

    assert asyncio.run(inside_loop()) == records

    path = bs.write_predictions(tmp_path / "predictions.jsonl", records)
    assert [json.loads(line) for line in path.read_text().splitlines()] == records
    with pytest.raises(FileExistsError):
        bs.write_predictions(path, records)
    with pytest.raises(ValueError, match="duplicate"):
        bs.write_predictions(tmp_path / "duplicate.jsonl", records + records[:1])
    with pytest.raises(ValueError):
        bs.write_predictions(tmp_path / "bad.jsonl", [{"id": pages[0].id, "view": "pdf", "injected": True}])


def test_cli_predict_writes_file(inputs, tmp_path, capsys, monkeypatch):
    (tmp_path / "leaderboard_detector.py").write_text(
        "def predict(request):\n    return 'Assistant note' in request.content\n")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "predictions.jsonl"
    cli.main(["predict", "--source", str(inputs), "--predict", "leaderboard_detector:predict",
              "--view", "html", "--view", "accessibility", "--output", str(output)])
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 2 * 6
    assert [r["view"] for r in records] == ["html"] * 6 + ["accessibility"] * 6
    summary = json.loads(capsys.readouterr().out)
    assert summary["revision"] is None and summary["dataset"]  # local source: no Hub revision
    assert summary["records"] == 12 and summary["views"]["html"] == {"pages": 6, "records": 6, "abstained": 0,
                                                                      "omitted": 0}

    testing.main([str(tmp_path / "trial"), "--test-inputs", "--pairs", "2"])
    assert len(bs.load_test_inputs(source=tmp_path / "trial")) == 2


def test_load_test_split_hub_error_mentions_space(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pq.write_table(pa.Table.from_pylist([row for row in rows(4) if row["split"] == "train"]),
                   data / "members-train-0000.parquet")
    from browsesafe_pp import dataset

    monkeypatch.setattr(dataset, "_files", lambda *args: (sorted(data.glob("*.parquet")), {}, "hub"))
    with pytest.raises(ValueError) as error:
        bs.load(split="test")
    message = str(error.value)
    assert "load_test_inputs" in message and bs.LEADERBOARD_SPACE in message and "coming soon" not in message


def test_predictions_follow_the_detector_on_published_pages(inputs, tmp_path):
    def detector(request):
        if "#2<" in request.content:
            raise RuntimeError("pair 2 fails")
        if "#3<" in request.content:
            return None
        return bs.Prediction("Assistant note" in request.content, score=0.75)

    paired = bs.load(source=write_fixture(tmp_path / "paired", n_pairs=6, shards=1), split=None)
    by_entry = {page.entry_id: page for page in paired}
    pages = bs.load_test_inputs(source=inputs, view="html")
    truth = labels(inputs)
    published = {truth[page.id]["entry_id"]: page for page in pages}
    assert len(published) == 6 and len({entry.split("_")[0] for entry in published}) == 6  # one page per pair
    assert all(page.content == by_entry[entry].content for entry, page in published.items())

    path = bs.write_predictions(tmp_path / "predictions.jsonl", bs.predict_test(pages, detector))
    records = {json.loads(line)["id"]: json.loads(line) for line in path.read_text().splitlines()}
    for entry, page in published.items():
        expected = None if "#2<" in page.content else ("abstain" if "#3<" in page.content else "Assistant note" in page.content)
        if expected is None:
            assert page.id not in records  # a failed call leaves the page out: an error at the Space
        elif expected == "abstain":
            assert records[page.id]["injected"] is None
        else:
            assert records[page.id] == {"id": page.id, "view": "html", "injected": expected, "score": 0.75}
            assert expected == truth[page.id]["injected"]  # the fixture detector reads the injection marker
