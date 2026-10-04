"""Optional Playwright browser target (``pip install 'browsesafe-pp[browser]'``).

Importing this package does not require Playwright; it is imported when a backend starts.
"""
from ..contracts import Action, Observation
from .snapshots import Delivery, RestoreReceipt, Snapshot, SnapshotStore
from .runtime import Backend, BrowserTarget, PlaywrightBackend
from .models import ChatModel, ModelError

__all__ = ["Action", "Backend", "BrowserTarget", "ChatModel", "Delivery", "ModelError", "Observation",
           "PlaywrightBackend", "RestoreReceipt", "Snapshot", "SnapshotStore"]
