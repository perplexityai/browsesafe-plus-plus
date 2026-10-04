"""Command line: ``browsesafe info`` and ``browsesafe evaluate``."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from . import __version__
from .dataset import ALL_VIEWS, DEFAULT_REVISION, describe, load
from .evaluation.classification import ModelDetector, evaluate, evaluate_async


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="browsesafe", description="BrowseSafe++ benchmark tools")
    parser.add_argument("--version", action="version", version=f"browsesafe-pp {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="Summarize a dataset source")
    _source_arguments(info)

    run = commands.add_parser("evaluate", help="Score a detector on attack/control pairs")
    _source_arguments(run)
    run.add_argument("--split", default="test", help="Partition to evaluate; 'all' for every partition")
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

    args = parser.parse_args(argv)
    if args.command == "info":
        summary = describe(source=args.source, dataset=args.dataset, revision=args.revision, offline=args.offline)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    benchmark = load(source=args.source, dataset=args.dataset, revision=args.revision, offline=args.offline,
                     split=None if args.split == "all" else args.split, view=args.view, seed=args.seed,
                     n_pairs=args.n_pairs or None)
    if args.predict:
        predict = _callable(args.predict)
        if asyncio.iscoroutinefunction(predict):
            report = asyncio.run(evaluate_async(benchmark, predict, concurrency=args.concurrency))
        else:
            report = evaluate(benchmark, predict)
    else:
        from .providers import OpenAICompatible

        async def run_model():
            provider = OpenAICompatible(endpoint=args.endpoint, model=args.model)
            detector = ModelDetector(provider, max_tokens=args.max_tokens)
            try:
                result = await evaluate_async(benchmark, detector, concurrency=args.concurrency)
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
