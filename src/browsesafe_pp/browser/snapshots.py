"""Local browser snapshot storage and delivery receipts."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from ..contracts import BrowserCheckpoint


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ValueError("Browser URLs must have an HTTP(S) origin without credentials")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = parsed.port
    suffix = f":{port}" if port and port != {"http": 80, "https": 443}[parsed.scheme] else ""
    return f"{parsed.scheme}://{host}{suffix}"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Delivery:
    """Receipt for the observation actually delivered at the selected tool call."""

    checkpoint_id: str
    tool_call_id: str
    observation_sha256: str
    view: str
    url: str
    tool_name: str



@dataclasses.dataclass(frozen=True)
class Snapshot:
    checkpoint: BrowserCheckpoint
    html: str
    storage_state: Mapping[str, Any]
    session_storage: Mapping[str, str]
    history: tuple[Mapping[str, Any], ...]
    app_state: Any
    scroll: tuple[float, float] = (0, 0)
    screenshot_path: str | None = None
    schema_version: int = 2
    screenshot_sha256: str | None = None
    content_sha256: str | None = None


@dataclasses.dataclass(frozen=True)
class RestoreReceipt:
    checkpoint_id: str
    snapshot_sha256: str
    adapter_version: str
    viewport: tuple[int, int]
    conversation_sha256: str
    application_state_sha256: str
    coverage: tuple[str, ...]
    checks: Mapping[str, bool | None]
    verified: bool | None


class SnapshotStore:
    """Local snapshots of one page, its conversation and supplied backend state.

    Cookies, local storage, IndexedDB and session storage are captured. Live JS
    heaps, other tabs, browser back stacks and network services require an
    application-specific restore hook; HTML serialization cannot restore them.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)

    def load(self, checkpoint_id: str) -> Snapshot:
        if not checkpoint_id or any(x not in "0123456789abcdef" for x in checkpoint_id):
            raise ValueError("Invalid snapshot identifier")
        record = json.loads((self.directory / f"{checkpoint_id}.json").read_text())
        expected = record.get("content_sha256")
        if expected is not None and expected != _digest({k: v for k, v in record.items()
                                                        if k != "content_sha256"}):
            raise ValueError("Snapshot content digest mismatch")
        if expected is None:
            record["schema_version"] = 1
        if record.get("screenshot_sha256") and record.get("screenshot_path"):
            if hashlib.sha256(Path(record["screenshot_path"]).read_bytes()).hexdigest() != record["screenshot_sha256"]:
                raise ValueError("Snapshot screenshot digest mismatch")
        checkpoint = record.pop("checkpoint")
        checkpoint["viewport"] = tuple(checkpoint["viewport"]) if checkpoint["viewport"] else None
        checkpoint["visible_history"] = tuple(checkpoint["visible_history"])
        record["history"] = tuple(record["history"])
        record["scroll"] = tuple(record["scroll"])
        return Snapshot(checkpoint=BrowserCheckpoint(**checkpoint), **record)

    def save_artifact(self, kind: str, record: Mapping[str, Any]) -> dict[str, str]:
        """Save an immutable evidence record (mode 600), returning its path and content digest."""
        if not kind or not all(x.isalnum() or x == "-" for x in kind):
            raise ValueError("Invalid artifact kind")
        path = self.directory / f"{kind}-{uuid.uuid4().hex}.json"
        encoded = json.dumps(dict(record), ensure_ascii=False, allow_nan=False).encode()
        with path.open("xb") as stream:
            path.chmod(0o600)
            stream.write(encoded)
        return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}

    async def capture(self, page: Any, *, history: tuple[Mapping[str, Any], ...] = (),
                      app_state: Any = None, tool_call_id: str | None = None,
                      tool_name: str = "browser.observe") -> Snapshot:
        _origin(page.url)
        checkpoint_id = uuid.uuid4().hex
        # Serialize live form values into the cloned DOM without altering the page.
        html = await page.evaluate("""() => {
            const root = document.documentElement.cloneNode(true);
            const live = document.querySelectorAll('input,textarea,option,details');
            const copied = root.querySelectorAll('input,textarea,option,details');
            live.forEach((node, i) => {
                const copy = copied[i];
                if (node.tagName === 'INPUT') {
                    if (node.type !== 'file') copy.setAttribute('value', node.value);
                    copy.toggleAttribute('checked', node.checked);
                } else if (node.tagName === 'TEXTAREA') copy.textContent = node.value;
                else if (node.tagName === 'OPTION') copy.toggleAttribute('selected', node.selected);
                else copy.toggleAttribute('open', node.open);
            });
            return '<!DOCTYPE html>\\n' + root.outerHTML;
        }""")
        viewport = page.viewport_size
        checkpoint = BrowserCheckpoint(
            checkpoint_id, tool_call_id or uuid.uuid4().hex, tool_name, page.url,
            (viewport["width"], viewport["height"]) if viewport else None, tuple(history))
        screenshot = self.directory / f"{checkpoint_id}.png"
        await page.screenshot(path=str(screenshot), full_page=False)
        screenshot.chmod(0o600)
        snapshot = Snapshot(
            checkpoint, html, await page.context.storage_state(indexed_db=True),
            await page.evaluate("() => Object.fromEntries(Object.entries(sessionStorage))"),
            tuple(history), app_state, tuple(await page.evaluate("() => [scrollX, scrollY]")),
            str(screenshot), screenshot_sha256=hashlib.sha256(screenshot.read_bytes()).hexdigest())
        record = dataclasses.asdict(snapshot)
        record.pop("content_sha256")
        snapshot = dataclasses.replace(snapshot, content_sha256=_digest(record))
        path = self.directory / f"{checkpoint_id}.json"
        with path.open("x", encoding="utf-8") as handle:
            path.chmod(0o600)
            json.dump(dataclasses.asdict(snapshot), handle, ensure_ascii=False, allow_nan=False)
        return snapshot
