"""Browser delivery adapter and configurable Playwright action loop."""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import math
import time
from typing import Any, Awaitable, Callable, Mapping, Protocol

from .. import contracts as a
from ..contracts import Action, Observation, TargetInput, TargetResult
from .snapshots import Delivery, RestoreReceipt, Snapshot, SnapshotStore, _digest, _origin


class Backend(Protocol):
    async def restore(self, checkpoint_id: str) -> None: ...

    async def deliver(self, request: TargetInput) -> Delivery: ...

    async def resume(self) -> TargetResult: ...

    async def aclose(self) -> None: ...


class BrowserTarget:
    """Restore, bind one page observation, then resume the agent's real rollout.

    A backend restores browser, conversation and application state and intercepts
    the selected tool result. For HTML, it renders the edited page and derives the
    views consumed by the agent. Receipts bind the supplied representation; view
    visibility and action outcomes belong in the rollout events and the judge.
    Construct a new backend for each target_factory call.
    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    async def run(self, request: TargetInput) -> TargetResult:
        try:
            return await self._run(request)
        except BaseException as exc:
            usage = self.usage_snapshot()
            if usage is not None:
                exc.usage = usage
            raise

    async def _run(self, request: TargetInput) -> TargetResult:
        checkpoint = request.checkpoint
        if checkpoint is None:
            raise ValueError("BrowserTarget requires a BrowserCheckpoint")
        await self.backend.restore(checkpoint.checkpoint_id)
        receipt = await self.backend.deliver(request)
        expected = Delivery(checkpoint.checkpoint_id, checkpoint.tool_call_id,
                            hashlib.sha256(request.observation.encode("utf-8")).hexdigest(),
                            request.view, checkpoint.url, checkpoint.tool_name)
        if not isinstance(receipt, Delivery) or receipt != expected:
            raise ValueError("Browser observation delivery does not match the checkpoint")
        result = await self.backend.resume()
        if not isinstance(result, TargetResult):
            raise TypeError("Browser backend must return TargetResult")
        event = {"type": "observation_delivered", **dataclasses.asdict(receipt)}
        return dataclasses.replace(result, events=(event, *result.events))

    async def aclose(self) -> None:
        await self.backend.aclose()

    def usage_snapshot(self) -> a.Usage | None:
        snapshot = getattr(self.backend, "usage_snapshot", None)
        return snapshot() if snapshot else None


class PlaywrightBackend:
    """Fresh browser context per rollout, using real actions and a supplied model.

    The edited HTML is served at the checkpoint URL; every observation view is
    derived from that page. Judges can inspect ``page`` and the restored
    application's state before ``aclose``. Application state callbacks must reset
    external services between trials. No provider calls are retried internally.
    """

    def __init__(self, store: SnapshotStore, model: Callable[[Observation], Awaitable[Action]], *,
                 allowed_origins: tuple[str, ...], view: str = "accessibility",
                 viewport: tuple[int, int] | None = None, max_steps: int = 20,
                 max_seconds: float = 120, action_timeout: float = 15,
                 restore_app: Callable[[Any], Awaitable[None]] | None = None,
                 snapshot_app: Callable[[], Awaitable[Any]] | None = None,
                 restore_page: Callable[[Any, Snapshot], Awaitable[None]] | None = None,
                 verify_restore: Callable[[Any, Snapshot], Awaitable[Mapping[str, bool | None]]] | None = None,
                 restoration_coverage: tuple[str, ...] = ("serialized_dom", "browser_storage", "session_storage", "conversation", "scroll"),
                 adapter_version: str = "playwright-html-v2",
                 collect_checkpoints: bool = False, headless: bool = True) -> None:
        if view not in {"html", "accessibility", "screenshot"}:
            raise ValueError("Choose html, accessibility or screenshot")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        if any(not math.isfinite(x) or x <= 0 for x in (max_seconds, action_timeout)):
            raise ValueError("Timeouts must be finite and positive")
        if viewport is not None and (len(viewport) != 2 or any(type(x) is not int or x <= 0 for x in viewport)):
            raise ValueError("viewport requires positive integer width and height")
        if not allowed_origins:
            raise ValueError("At least one application origin is required")
        self.store, self.model = store, model
        self.allowed_origins = frozenset(_origin(url) for url in allowed_origins)
        self.view, self.viewport, self.max_steps = view, viewport, max_steps
        self.max_seconds, self.action_timeout = max_seconds, action_timeout
        self.restore_app, self.snapshot_app, self.restore_page = restore_app, snapshot_app, restore_page
        self.verify_restore = verify_restore
        self.restoration_coverage = tuple(restoration_coverage)
        self.adapter_version = adapter_version
        self.restore_receipt: RestoreReceipt | None = None
        self.collect_checkpoints, self.headless = collect_checkpoints, headless
        self.page = self.context = self.browser = self.playwright = None
        self.snapshot: Snapshot | None = None
        self.history: list[Mapping[str, Any]] = []
        self.events: list[Mapping[str, Any]] = []
        self.task = ""
        self._deadline = 0.0
        self._usage: list[a.Usage] = []
        self._delivered = False
        self.request: a.TargetInput | None = None
        self._session_script = self._cdp = None

    async def restore(self, checkpoint_id: str) -> None:
        if self.playwright is not None:
            raise RuntimeError("Construct a fresh backend for each rollout")
        from playwright.async_api import async_playwright
        self.snapshot = self.store.load(checkpoint_id)
        if _origin(self.snapshot.checkpoint.url) not in self.allowed_origins:
            raise ValueError("Checkpoint origin is outside the application allowlist")
        if self.snapshot.app_state is not None and self.restore_app is None:
            raise ValueError("Snapshot has application state but no restore_app callback")
        if self.restore_app:
            await self.restore_app(self.snapshot.app_state)
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        size = self.viewport or self.snapshot.checkpoint.viewport or (1280, 800)
        self.context = await self.browser.new_context(
            storage_state=dict(self.snapshot.storage_state),
            viewport={"width": size[0], "height": size[1]}, service_workers="block")
        session_data = json.dumps(dict(self.snapshot.session_storage), ensure_ascii=True)
        origin_data = json.dumps(_origin(self.snapshot.checkpoint.url))
        self.page = await self.context.new_page()
        # Remove this initializer after the first document so later navigation
        # preserves session changes made during the rollout.
        self._cdp = await self.context.new_cdp_session(self.page)
        await self._cdp.send("Page.enable")
        self._session_script = (await self._cdp.send("Page.addScriptToEvaluateOnNewDocument", {
            "source": f"if (location.origin === {origin_data}) {{ for (const [k,v] of Object.entries({session_data})) sessionStorage.setItem(k,v); }}"
        }))["identifier"]
        self.page.set_default_timeout(self.action_timeout * 1000)
        self.history = list(self.snapshot.history)
        self._deadline = time.monotonic() + self.max_seconds

    async def deliver(self, request: a.TargetInput) -> Delivery:
        if self.snapshot is None or self.page is None or self._delivered:
            raise RuntimeError("Restore once before delivering one observation")
        checkpoint = self.snapshot.checkpoint
        if request.checkpoint != checkpoint or request.view != "html":
            raise ValueError("Runtime delivery requires the original checkpoint and HTML")
        self.task = request.task
        self.request = request
        delivered = False

        async def route_request(route: Any) -> None:
            nonlocal delivered
            url = route.request.url
            try:
                allowed = _origin(url) in self.allowed_origins
            except ValueError:
                allowed = False
            if not allowed:
                self.events.append({"type": "network_blocked", "url": url})
                await route.abort()
            elif not delivered and url.split("#")[0] == checkpoint.url.split("#")[0] and route.request.is_navigation_request():
                delivered = True
                await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=request.observation)
            else:
                await route.continue_()

        await self.context.route("**/*", route_request)
        try:
            await self.page.goto(checkpoint.url, wait_until="domcontentloaded")
        finally:
            await self._cdp.send("Page.removeScriptToEvaluateOnNewDocument", {"identifier": self._session_script})
            self._session_script = None
        if not delivered:
            raise RuntimeError("Checkpoint HTML was not delivered")
        if self.restore_page:
            await self.restore_page(self.page, self.snapshot)
        await self.page.evaluate("([x,y]) => scrollTo(x,y)", list(self.snapshot.scroll))
        size = self.page.viewport_size
        actual_session = await self.page.evaluate("() => Object.fromEntries(Object.entries(sessionStorage))")
        checks: dict[str, bool | None] = {
            "session_storage": actual_session == dict(self.snapshot.session_storage),
            "conversation": tuple(self.history) == tuple(self.snapshot.history),
            "viewport": (size["width"], size["height"]) == (self.viewport or checkpoint.viewport or (1280, 800)),
            "task_state": None,
        }
        if self.snapshot.app_state is not None and self.snapshot_app:
            checks["application_state"] = _digest(await self.snapshot_app()) == _digest(self.snapshot.app_state)
        elif self.snapshot.app_state is not None:
            checks["application_state"] = None
        if self.verify_restore:
            custom = await self.verify_restore(self.page, self.snapshot)
            if not isinstance(custom, Mapping) or any(not isinstance(k, str) or
                    (v is not None and type(v) is not bool) for k, v in custom.items()):
                raise ValueError("verify_restore must return named bool/None checks")
            checks.update({f"task:{key}": value for key, value in custom.items()})
            checks["task_state"] = (False if False in custom.values() else
                                    True if custom and all(v is True for v in custom.values()) else None)
        verified = False if False in checks.values() else (True if all(v is True for v in checks.values()) else None)
        self.restore_receipt = RestoreReceipt(
            checkpoint.checkpoint_id, self.snapshot.content_sha256 or _digest(dataclasses.asdict(self.snapshot)),
            self.adapter_version, (size["width"], size["height"]), _digest(self.history),
            _digest(self.snapshot.app_state), self.restoration_coverage, checks, verified)
        record = dataclasses.asdict(self.restore_receipt)
        artifact = self.store.save_artifact("restore", record)
        self.events.append({"type": "state_restored", **record, "artifact": artifact})
        if verified is False:
            raise ValueError("Restored state failed verification")
        self._delivered = True
        return Delivery(checkpoint.checkpoint_id, checkpoint.tool_call_id,
                        hashlib.sha256(request.observation.encode()).hexdigest(),
                        request.view, checkpoint.url, checkpoint.tool_name)

    async def _observe(self, step: int) -> Observation:
        if self.view == "html":
            content, screenshot = await self.page.content(), None
        elif self.view == "accessibility":
            content, screenshot = await self.page.locator("body").aria_snapshot(), None
        else:
            content = ""
            screenshot = base64.b64encode(await self.page.screenshot(full_page=False)).decode("ascii")
        size = self.page.viewport_size
        self.events.append({"type": "agent_observation", "step": step, "url": self.page.url,
                            "view": self.view, "viewport": size,
                            "content_encoding": "base64_png" if screenshot else "utf8",
                            "content_sha256": hashlib.sha256((screenshot or content).encode()).hexdigest()})
        if self.collect_checkpoints:
            state = await self.snapshot_app() if self.snapshot_app else None
            saved = await self.store.capture(self.page, history=tuple(self.history), app_state=state)
            self.events.append({"type": "checkpoint_saved", "checkpoint_id": saved.checkpoint.checkpoint_id,
                                "step": step})
        usage = self._total_usage()
        token_cap = self.request.max_tokens
        token_left = (token_cap - usage.input_tokens - usage.output_tokens
                      if token_cap is not None and usage.input_tokens is not None
                      and usage.output_tokens is not None else None)
        cost_left = (self.request.max_cost_usd - usage.cost_usd
                     if self.request.max_cost_usd is not None and usage.cost_usd is not None else None)
        observation = Observation(self.task, self.page.url, self.view, content, screenshot,
                           tuple(self.history), (size["width"], size["height"]), step,
                           self.request.trial_seed, token_left, cost_left)
        artifact = self.store.save_artifact("observation", dataclasses.asdict(observation))
        self.events.append({"type": "observation_saved", "step": step, "artifact": artifact})
        return observation

    async def _act(self, action: Action) -> str | None:
        args = action.arguments
        if action.name == "finish":
            if not isinstance(args.get("text"), str):
                raise ValueError("finish requires text")
            return args["text"]
        if action.name in {"click", "fill"}:
            if action.name == "click" and "selector" not in args:
                if any(type(args.get(key)) not in {int, float} or not math.isfinite(args[key])
                       for key in ("x", "y")):
                    raise ValueError("Coordinate click requires finite x and y")
                await self.page.mouse.click(args["x"], args["y"])
                return None
            if not isinstance(args.get("selector"), str) or not args["selector"]:
                raise ValueError("click/fill requires a CSS selector")
            locator = self.page.locator(args["selector"])
            if action.name == "click":
                await locator.click()
            else:
                if not isinstance(args.get("text"), str):
                    raise ValueError("fill requires text")
                await locator.fill(args["text"])
        elif action.name in {"type", "press"}:
            key = "text" if action.name == "type" else "key"
            if not isinstance(args.get(key), str):
                raise ValueError(f"{action.name} requires {key}")
            if action.name == "type":
                await self.page.keyboard.insert_text(args["text"])
            else:
                await self.page.keyboard.press(args["key"])
        elif action.name == "scroll":
            x, y = args.get("x", 0), args.get("y", 600)
            if any(type(v) not in {int, float} or not math.isfinite(v) for v in (x, y)):
                raise ValueError("scroll coordinates must be finite numbers")
            await self.page.evaluate("([x,y]) => scrollBy(x,y)", [x, y])
        elif action.name == "goto":
            if not isinstance(args.get("url"), str) or _origin(args["url"]) not in self.allowed_origins:
                raise ValueError("Navigation outside allowed application origins")
            await self.page.goto(args["url"], wait_until="domcontentloaded")
        elif action.name == "back":
            await self.page.go_back(wait_until="domcontentloaded")
        return None

    def _total_usage(self) -> a.Usage:
        return a.Usage.aggregate(self._usage)

    def usage_snapshot(self) -> a.Usage:
        return self._total_usage()

    async def resume(self) -> a.TargetResult:
        try:
            return await self._resume()
        except BaseException as exc:
            exc.usage = self._total_usage()
            self.events.append({"type": "rollout_interrupted", "error_type": type(exc).__name__,
                                "usage": dataclasses.asdict(exc.usage)})
            self.store.save_artifact("interrupted", {"events": self.events, "usage": dataclasses.asdict(exc.usage)})
            raise

    async def _resume(self) -> a.TargetResult:
        if not self._delivered:
            raise RuntimeError("Deliver an observation before resuming")
        output, reason = "", "step_limit"
        for step in range(self.max_steps):
            usage = self._total_usage()
            if self.request.max_model_calls is not None and usage.model_calls >= self.request.max_model_calls:
                reason = "model_call_limit"
                break
            if self.request.max_tokens is not None and (usage.input_tokens is None
                    or usage.output_tokens is None
                    or usage.input_tokens + usage.output_tokens >= self.request.max_tokens):
                reason = "token_limit_or_unknown"
                break
            if self.request.max_cost_usd is not None and (usage.cost_usd is None
                    or usage.cost_usd >= self.request.max_cost_usd):
                reason = "cost_limit_or_unknown"
                break
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                reason = "time_limit"
                break
            observation = await asyncio.wait_for(self._observe(step), remaining)
            # Count dispatch before awaiting the model; failed requests still cost a call.
            self._usage.append(a.Usage(model_calls=1, input_tokens=None, output_tokens=None, cost_usd=None))
            try:
                action = await asyncio.wait_for(self.model(observation), max(0.001, self._deadline - time.monotonic()))
                if not isinstance(action, Action):
                    raise TypeError("Model must return an Action")
            except BaseException as exc:
                reported = getattr(exc, "usage", None)
                if isinstance(reported, a.Usage):
                    self._usage[-1] = reported
                exc.usage = self._total_usage()
                raise
            if action.usage is not None:
                self._usage[-1] = action.usage
            usage = self._total_usage()
            overrun = {}
            for cap, value in (("max_model_calls", usage.model_calls),
                               ("max_tokens", usage.input_tokens + usage.output_tokens
                                if usage.input_tokens is not None and usage.output_tokens is not None else None),
                               ("max_cost_usd", usage.cost_usd)):
                limit = getattr(self.request, cap)
                if limit is not None and value is not None and value > limit:
                    overrun[cap] = value - limit
            if overrun:
                self.events.append({"type": "usage_limit_overshoot", "step": step, "overrun": overrun})
            event = {"type": "browser_action", "step": step, "name": action.name,
                     "arguments": dict(action.arguments), "url_before": self.page.url}
            try:
                finished = await asyncio.wait_for(self._act(action), min(self.action_timeout, max(
                    0.001, self._deadline - time.monotonic())))
                event.update(status="completed", url_after=self.page.url)
            except Exception as exc:
                event.update(status="error", error=f"{type(exc).__name__}: {exc}")
                finished = None
            self.events.append(event)
            self.store.save_artifact("action", {**event, "usage": dataclasses.asdict(self._usage[-1])})
            self.history.append({"role": "assistant", "action": action.name,
                                 "arguments": dict(action.arguments), "result": dict(event)})
            if finished is not None:
                output, reason = finished, "finished"
                break
        self.events.append({"type": "rollout_stopped", "reason": reason})
        return a.TargetResult(output, events=tuple(self.events), usage=self._total_usage())

    async def aclose(self) -> None:
        try:
            if self.context:
                await self.context.close()
        finally:
            try:
                if self.browser:
                    await self.browser.close()
            finally:
                if self.playwright:
                    await self.playwright.stop()
