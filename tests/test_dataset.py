import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import browsesafe_pp as bs
from browsesafe_pp.testing import rows


def test_default_dataset_and_overrides(monkeypatch):
    monkeypatch.delenv("BROWSESAFE_DATASET", raising=False)
    assert bs.dataset_id() == "perplexity-ai/browsesafe-plus-plus"
    monkeypatch.setenv("BROWSESAFE_DATASET", "example/alternate")
    assert bs.dataset_id() == "example/alternate"
    assert bs.dataset_id("example/explicit") == "example/explicit"


def test_hub_test_split_is_reserved(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    train = [row for row in rows(4) if row["split"] == "train"]
    pq.write_table(pa.Table.from_pylist(train), data / "members-train-0000.parquet")
    from browsesafe_pp import dataset

    monkeypatch.setattr(dataset, "_files", lambda *args: (sorted(data.glob("*.parquet")), {}, "hub"))
    assert bs.load().manifest["loaded_pairs"] == len(train) // 2
    with pytest.raises(ValueError, match="test split is reserved for the leaderboard"):
        bs.load(split="test")


def test_split_shards_with_image_column(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    records = rows(4)
    for split in ("train", "test"):
        selected = [dict(row, screenshot={"bytes": None, "path": "https://example.com/image.jpg"})
                    for row in records if row["split"] == split]
        pq.write_table(pa.Table.from_pylist(selected), data / f"members-{split}-0000.parquet")
    benchmark = bs.load(source=tmp_path, split="test", view="markdown")
    assert len(benchmark.pairs) == 2
    assert all(pair.metadata["split"] == "test" for pair in benchmark.pairs)
    info = bs.describe(source=tmp_path)
    assert info["pages"] == 8 and info["pairs_by_split"] == {"train": 2, "test": 2}


def test_load_pairs_and_labels(source):
    benchmark = bs.load(source=source, split="test")
    assert benchmark.manifest["loaded_pairs"] == 6
    assert len(benchmark) == 12
    for pair in benchmark.pairs:
        assert pair.malicious.entry_id == f"{pair.pair_id}_malicious"
        assert pair.benign.entry_id == f"{pair.pair_id}_benign"
        assert pair.malicious.injected and not pair.benign.injected
        assert pair.malicious.metadata["split"] == pair.benign.metadata["split"] == "test"
    pages = list(benchmark)
    assert [p.injected for p in pages[:2]] == [True, False]


def test_views_and_evidence(source):
    for view, marker in (("html", "<main>"), ("markdown", "# "), ("accessibility", "- main:")):
        benchmark = bs.load(source=source, view=view, n_pairs=2)
        page = benchmark.pairs[0].malicious
        assert marker in page.content and page.view == view
        assert page.evidence and all(page.content[s:e] in page.evidence for s, e in page.evidence_spans)
        assert benchmark.pairs[0].benign.evidence == ()
    assert "content" in bs.load(source=source, n_pairs=1).pairs[0].malicious.model_input.__dict__


def test_sampling_is_reproducible_and_seeded(source):
    first = bs.load(source=source, split=None, n_pairs=5, seed=1).manifest["pair_ids"]
    again = bs.load(source=source, split=None, n_pairs=5, seed=1).manifest["pair_ids"]
    other = bs.load(source=source, split=None, n_pairs=12, seed=2).manifest["pair_ids"]
    assert first == again and len(first) == 5
    assert sorted(other) == sorted(bs.load(source=source, split=None).manifest["pair_ids"])


def test_filters(source):
    articles = bs.load(source=source, split=None, filters={"page_type": "article"})
    assert articles.pairs and all(p.metadata["page_type"] == "article" for p in articles.pairs)
    spanish = bs.load(source=source, split=None, filters={"language_page": ["es"]})
    assert spanish.pairs and all("es" in p.metadata["language_page"] for p in spanish.pairs)
    with pytest.raises(ValueError):
        bs.load(source=source, filters={"not_a_field": 1})


def test_missing_split_and_bad_view(source):
    with pytest.raises(ValueError, match="absent"):
        bs.load(source=source, split="validation")
    with pytest.raises(ValueError):
        bs.load(source=source, view="pdf")


def test_incomplete_and_inconsistent_pairs(tmp_path):
    records = rows(4)
    records = [r for r in records if r["entry_id"] != "00001_benign"]  # pair 1 incomplete
    records[2]["label"] = "yes"  # 00002_benign mislabeled -> invalid pair
    (tmp_path / "data").mkdir()
    pq.write_table(pa.Table.from_pylist(records), tmp_path / "data" / "members-00000.parquet")
    benchmark = bs.load(source=tmp_path, split=None)
    assert benchmark.manifest["exclusions"]["incomplete_pairs"] == 1
    assert benchmark.manifest["exclusions"]["invalid_pairs"] == 1
    assert sorted(benchmark.manifest["pair_ids"]) == ["00003", "00004"]


def test_missing_view_excludes_pair(tmp_path):
    records = rows(3)
    records[0]["content_markdown"] = None
    (tmp_path / "data").mkdir()
    pq.write_table(pa.Table.from_pylist(records), tmp_path / "data" / "members-00000.parquet")
    benchmark = bs.load(source=tmp_path, split=None, view="markdown")
    assert benchmark.manifest["loaded_pairs"] == 2
    assert benchmark.manifest["exclusions"]["missing_view_pairs"] == 1


@pytest.mark.parametrize("order", ["selection", "source"])
def test_iter_batches_matches_load(source, order, tmp_path):
    loaded = bs.load(source=source, split=None)
    batches = list(bs.iter_batches(source=source, split=None, batch_pairs=5, order=order, spool_dir=tmp_path))
    ids = [pair.pair_id for batch in batches for pair in batch.pairs]
    assert sorted(ids) == sorted(loaded.manifest["pair_ids"])
    if order == "selection":
        assert ids == loaded.manifest["pair_ids"]
    assert batches[-1].manifest["batch"]["is_final"]
    assert not list(tmp_path.iterdir())  # spool cleaned up


def test_describe(source):
    summary = bs.describe(source=source)
    assert summary["pairs"] == 12 and summary["pages"] == 24
    assert summary["pairs_by_split"] == {"test": 6, "train": 6}
    assert summary["text_views"] == ["html", "markdown", "accessibility"]


def test_records_and_dataset_id(source, monkeypatch):
    records = list(bs.load(source=source, n_pairs=1).to_records())
    assert [r["label"] for r in records] == ["yes", "no"]
    monkeypatch.setenv("BROWSESAFE_DATASET", "someone/else")
    assert bs.dataset_id() == "someone/else"
    assert bs.dataset_id("x/y") == "x/y"
    monkeypatch.delenv("BROWSESAFE_DATASET")
    assert bs.dataset_id() == bs.DEFAULT_DATASET


def test_back_compat_imports():
    from browsesafe_pp.dataset import evaluate, DetectionReport  # noqa: F401
    assert evaluate is bs.evaluate
