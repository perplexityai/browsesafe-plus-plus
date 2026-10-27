"""Command line: ``browsesafe info``, ``browsesafe evaluate`` and ``browsesafe predict``."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
from pathlib import Path

from . import __version__
from .dataset import ALL_VIEWS, DEFAULT_REVISION, _resolve_revision, dataset_id, describe, load
from .evaluation.classification import ModelDetector, evaluate, evaluate_async
from .test_inputs import TEST_DIRECTORY, TEST_SHARD_PATTERN, load_test_inputs, predict_test_async, write_predictions


def _source_arguments(parser: argparse.ArgumentParser, shards: str = "members-*.parquet") -> None:
    parser.add_argument("--source", type=Path, help=f"Local directory of {shards} shards")
    parser.add_argument("--dataset", help="Hugging Face dataset id (default: $BROWSESAFE_DATASET, else "
                                          "perplexity-ai/browsesafe-plus-plus)")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Dataset branch, tag or commit")
    parser.add_argument("--offline", action="store_true", help="Use the cached snapshot only")


def _callable(spec: str):
    module, separator, name = spec.partition(":")
    if not separator or not name:
        raise SystemExit("--predict must look like module:function")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module), name)


def _predict(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; choose a new file")
    predict = _callable(args.predict)
    # Resolve the revision once, so every view reads the same commit and the summary can report it (the licence asks for it).
    repo_id = dataset_id(args.dataset)
    revision = args.revision if args.source is not None else _resolve_revision(repo_id, args.revision, None,
                                                                                args.offline)

    async def run():
        records, views = [], {}
        for view in dict.fromkeys(args.view):
            pages = load_test_inputs(view=view, source=args.source, dataset=args.dataset, revision=revision,
                                     offline=args.offline, limit=args.limit)
            written = await predict_test_async(pages, predict, concurrency=args.concurrency)
            views[view] = {"pages": len(pages), "records": len(written),
                           "abstained": sum(record["injected"] is None for record in written),
                           "omitted": len(pages) - len(written)}
            records.extend(written)
        return records, views

    records, views = asyncio.run(run())
    path = write_predictions(args.output, records)
    dataset = str(args.source) if args.source is not None else repo_id
    print(json.dumps({"output": str(path), "dataset": dataset, "revision": revision if args.source is None else None,
                      "records": len(records), "views": views}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="browsesafe", description="BrowseSafe++ benchmark tools")
    parser.add_argument("--version", action="version", version=f"browsesafe-pp {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="Summarize a dataset source")
    _source_arguments(info)

    run = commands.add_parser("evaluate", help="Score a detector on attack/control pairs")
    _source_arguments(run)
    run.add_argument("--split", default="train", help="Partition to evaluate (the Hub dataset's labelled pairs are 'train'); 'all' for every partition")
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

    submit = commands.add_parser("predict", help="Write leaderboard predictions for the test inputs")
    _source_arguments(submit, shards=f"{TEST_DIRECTORY}/data/{TEST_SHARD_PATTERN}")
    submit.add_argument("--predict", required=True, help="Detector as module:function (sync or async)")
    submit.add_argument("--view", choices=ALL_VIEWS, action="append", required=True,
                        help="View to predict; repeat the option for several views")
    submit.add_argument("--output", type=Path, required=True, help="New JSON Lines file to upload to the Space")
    submit.add_argument("--limit", type=int, help="Only the first N pages of each view, for a trial run")
    submit.add_argument("--concurrency", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "predict":
        _predict(args)
        return
    if args.command == "info":
        summary = describe(source=args.source, dataset=args.dataset, revision=args.revision, offline=args.offline)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    benchmark = load(source=args.source, dataset=args.dataset, revision=args.revision, offline=args.offline,
                     split=None if args.split == "all" else args.split, view=args.view, seed=args.seed,
                     n_pairs=args.n_pairs or None)
    if args.predict:
        predict = _callable(args.predict)
        if inspect.iscoroutinefunction(predict):
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
