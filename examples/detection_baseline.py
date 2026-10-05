"""Score a simple keyword baseline (not a defense) on BrowseSafe++ pairs.

    python -m browsesafe_pp.testing ./fixture          # offline demo data
    python examples/detection_baseline.py --source ./fixture
"""
import argparse
import json

import browsesafe_pp as bs

INDICATORS = ("ignore previous", "ignore all previous", "disregard previous", "system prompt",
              "assistant note", "ai assistant", "you are now")


def predict(request: bs.DetectorInput) -> bs.Prediction:
    text = request.content.lower()
    hits = sum(word in text for word in INDICATORS)
    return bs.Prediction(injected=hits > 0, score=min(1.0, hits / 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", help="Local shard directory; omit to download from the Hub")
    parser.add_argument("--split", default="train",
                        help="Partition; 'test' is private and only available in a local source or the internal dataset")
    parser.add_argument("--view", default="html", choices=("html", "markdown", "accessibility"))
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--output", help="Optional new directory for the report")
    args = parser.parse_args()
    benchmark = bs.load(source=args.source, split=args.split, view=args.view, n_pairs=args.n_pairs)
    report = bs.evaluate(benchmark, predict)
    if args.output:
        report.save(args.output)
    print(json.dumps(report.metrics, indent=2))


if __name__ == "__main__":
    main()
