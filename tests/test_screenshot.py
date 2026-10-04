import shutil

import pytest

import browsesafe_pp as bs


@pytest.mark.parametrize("bucketed", [False, True])
def test_screenshot_view(source, tmp_path, bucketed):
    Image = pytest.importorskip("PIL.Image")
    root = tmp_path / "src"
    shutil.copytree(source, root)
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

    from browsesafe_pp import screenshot
    parts = screenshot.user_content(benchmark.pairs[0].malicious.content, tile_height=1536, max_tiles=1)
    assert parts[0]["type"] == "text" and "not shown" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
