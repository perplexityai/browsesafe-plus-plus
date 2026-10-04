"""Screenshot view: a full-page JPEG per page, presented to the model as a bounded set of readable segments.

The text views are parquet columns; the screenshot view is a directory of JPEG files,
``screenshots/<bucket>/<entry_id>.jpg`` in the dataset repository (downloaded with the shards)
or ``<source>/screenshots/<entry_id>.jpg`` in a local source. A page's screenshot is split into
segments of ``tile_height`` pixels at its stored width, at most ``max_tiles`` segments from the top;
what lies below the last segment is reported to the model as omitted, never silently dropped.
Requires Pillow (``pip install 'browsesafe-pp[screenshot] @ git+https://github.com/perplexityai/browsesafe-plus-plus.git'``).
"""
from __future__ import annotations

import base64, functools, io, os
from pathlib import Path

TILE_HEIGHT = 1536
MAX_TILES = 8
QUALITY = 80


@functools.lru_cache(maxsize=256)
def _prepared(path: str, mtime: float, tile_height: int, max_tiles: int, quality: int):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        tiles = []
        for top in range(0, min(height, tile_height * max_tiles), tile_height):
            tile = image.crop((0, top, width, min(height, top + tile_height)))
            buffer = io.BytesIO()
            tile.save(buffer, "JPEG", quality=quality, optimize=True)
            tiles.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    covered = min(height, tile_height * max_tiles)
    return tuple(tiles), {"width": width, "height": height, "covered_height": covered,
                          "omitted_height": height - covered, "segments": len(tiles)}


def tiles(path: str | Path, *, tile_height: int = TILE_HEIGHT, max_tiles: int = MAX_TILES, quality: int = QUALITY):
    """Base64 JPEG segments (top to bottom) and a description of the coverage."""
    path = str(path)
    return _prepared(path, os.stat(path).st_mtime, tile_height, max_tiles, quality)


def user_content(path: str | Path, **options) -> list[dict]:
    """OpenAI-style content parts: one text part describing the capture, then the image segments."""
    segments, info = tiles(path, **options)
    note = (f"Full-page screenshot of the webpage, {info['segments']} segment(s) from top to bottom, "
            f"{info['width']}x{info['covered_height']} pixels shown")
    if info["omitted_height"]:
        note += f"; the lowest {info['omitted_height']} pixels of the page are not shown"
    return [{"type": "text", "text": note + "."}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}} for data in segments]
