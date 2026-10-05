"""Command line: ``browsesafe info``, ``browsesafe evaluate`` and ``browsesafe serve``."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .dataset import ALL_VIEWS, DEFAULT_REVISION, DetectorInput, describe, load
from .evaluation.classification import ModelDetector, _outcome, evaluate, evaluate_async

MAX_REQUEST_BYTES = 256 * 1024 * 1024


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, help="Local directory of members-*.parquet shards")
    parser.add_argument("--dataset", help="Hugging Face dataset id (default: $BROWSESAFE_DATASET)")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Dataset branch, tag or commit")
    parser.add_argument("--offline", action="store_true", help="Use the cached snapshot only")


def _callable(spec: str):
    module, separator, name = spec.partition(":")
    if not separator or not name:
        raise SystemExit("--predict must look like module:function")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module), name)


def make_server(predict: Callable[..., Any], host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """An HTTP server exposing ``predict`` under the leaderboard endpoint contract.

    ``POST /predict`` takes ``{"content": str, "view": str, "task": str | null}`` and returns
    ``{"injected": bool | null, "score": float | null}``; ``GET /health`` returns the SDK version.
    A detector exception answers ``500`` and an invalid return value ``422``, so
    :class:`~browsesafe_pp.EndpointDetector` scores them as ``error`` and ``invalid_prediction``,
    exactly like ``browsesafe evaluate`` does for a local function. ``port=0`` picks a free port.
    """
    if not callable(predict):
        raise TypeError("predict must be callable")
    is_async = inspect.iscoroutinefunction(predict)

    class Handler(BaseHTTPRequestHandler):
        server_version = f"browsesafe-pp/{__version__}"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path.split("?", 1)[0] != "/health":
                return self._send(404, {"error": "not_found"})
            self._send(200, {"status": "ok", "sdk_version": __version__})

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if self.path.split("?", 1)[0] != "/predict":
                return self._send(404, {"error": "not_found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._send(400, {"error": "invalid_content_length"})
            if length <= 0 or length > MAX_REQUEST_BYTES:
                return self._send(413 if length > MAX_REQUEST_BYTES else 411, {"error": "invalid_content_length"})
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                return self._send(400, {"error": "invalid_json"})
            if (not isinstance(body, dict) or not isinstance(body.get("content"), str)
                    or not isinstance(body.get("view"), str)
                    or (body.get("task") is not None and not isinstance(body["task"], str))):
                return self._send(400, {"error": "invalid_request",
                                        "expected": {"content": "str", "view": "str", "task": "str | null"}})
            request = DetectorInput(content=body["content"], view=body["view"], task=body.get("task"))
            try:
                result = asyncio.run(predict(request)) if is_async else predict(request)
            except Exception as exc:  # the detector failed; never report a label
                return self._send(500, {"error": "detector_error", "type": type(exc).__name__})
            status, injected, score = _outcome(result)
            if status == "invalid_prediction":
                return self._send(422, {"error": "invalid_prediction"})
            self._send(200, {"injected": injected, "score": score})

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="browsesafe", description="BrowseSafe++ benchmark tools")
    parser.add_argument("--version", action="version", version=f"browsesafe-pp {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="Summarize a dataset source")
    _source_arguments(info)

    run = commands.add_parser("evaluate", help="Score a detector on attack/control pairs")
    _source_arguments(run)
    run.add_argument("--split", default="train",
                     help="Partition to evaluate; 'all' for every partition. 'test' is private (docs/leaderboard.md)")
    run.add_argument("--view", choices=ALL_VIEWS, default="html")
    run.add_argument("--n-pairs", type=int, default=100, help="Number of pairs; 0 for all")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", type=Path, help="New directory for manifest, metrics and predictions")
    detector = run.add_mutually_exclusive_group(required=True)
    detector.add_argument("--predict", help="Detector as module:function (sync or async)")
    detector.add_argument("--model", help="Model name for the OpenAI-compatible provider")
    run.add_argument("--endpoint", help="Provider base URL (default: $BROWSESAFE_MODEL_ENDPOINT)")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--evaluator", default="self", help="Who ran the evaluation, recorded in the manifest")
    run.add_argument("--run-id", help="Optional run identifier, recorded in the manifest")

    serve = commands.add_parser("serve", help="Serve a detector as POST /predict for leaderboard evaluation")
    serve.add_argument("--predict", required=True, help="Detector as module:function (sync or async)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "info":
        summary = describe(source=args.source, dataset=args.dataset, revision=args.revision, offline=args.offline)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if args.command == "serve":
        server = make_server(_callable(args.predict), args.host, args.port)
        host, port = server.server_address[:2]
        print(f"Serving {args.predict} at http://{host}:{port}/predict (GET /health)", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return

    benchmark = load(source=args.source, dataset=args.dataset, revision=args.revision, offline=args.offline,
                     split=None if args.split == "all" else args.split, view=args.view, seed=args.seed,
                     n_pairs=args.n_pairs or None)
    if args.predict:
        predict = _callable(args.predict)
        if inspect.iscoroutinefunction(predict):
            report = asyncio.run(evaluate_async(benchmark, predict, concurrency=args.concurrency,
                                                evaluator=args.evaluator, run_id=args.run_id))
        else:
            report = evaluate(benchmark, predict, evaluator=args.evaluator, run_id=args.run_id)
    else:
        from .providers import OpenAICompatible

        async def run_model():
            provider = OpenAICompatible(endpoint=args.endpoint, model=args.model)
            detector = ModelDetector(provider, max_tokens=args.max_tokens)
            try:
                result = await evaluate_async(benchmark, detector, concurrency=args.concurrency,
                                              evaluator=args.evaluator, run_id=args.run_id)
            finally:
                await provider.aclose()
            result.manifest["detector"] = detector.configuration
            return result

        report = asyncio.run(run_model())
    if args.output:
        report.save(args.output)
    print(json.dumps(report.metrics, indent=2))


if __name__ == "__main__":
    main()
