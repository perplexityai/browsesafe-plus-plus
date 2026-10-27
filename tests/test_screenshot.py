import pytest

import browsesafe_pp as bs
from browsesafe_pp.testing import write_fixture


@pytest.mark.parametrize("bucketed", [False, True])
def test_screenshot_view_from_files(tmp_path, hf_home, bucketed):
    """Shards without embedded images (dataset v0.1, older copies) read the screenshot files."""
    Image = pytest.importorskip("PIL.Image")
    root = write_fixture(tmp_path / "src", n_pairs=12, embed_screenshots=False)
    shots = root / "screenshots"
    shots.mkdir()
    for pair_id in ("00001", "00002"):
        for arm in ("malicious", "benign"):
            folder = shots / pair_id[:2] if bucketed else shots
            folder.mkdir(exist_ok=True)
            Image.new("RGB", (400, 2000), "white").save(folder / f"{pair_id}_{arm}.jpg")
    benchmark = bs.load(source=root, split=None, view="screenshot")
    assert sorted(benchmark.manifest["pair_ids"]) == ["00001", "00002"]
    assert benchmark.manifest["exclusions"]["missing_view_pairs"] == 10
    assert all(page.content.startswith(str(shots.resolve())) for page in benchmark)
    assert not (hf_home / "browsesafe_pp").exists()

    from browsesafe_pp import screenshot
    parts = screenshot.user_content(benchmark.pairs[0].malicious.content, tile_height=1536, max_tiles=1)
    assert parts[0]["type"] == "text" and "not shown" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
