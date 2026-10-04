"""Evaluate an OpenAI-compatible model with the BrowseSafe 1.0 classification prompt.

    export BROWSESAFE_MODEL_ENDPOINT=https://api.openai.com/v1
    export BROWSESAFE_API_KEY=...            # if your endpoint needs one
    python examples/evaluate_model.py --model gpt-4o-mini --n-pairs 50

Add ``--mock`` to run fully offline against a stand-in model (useful with ``--source ./fixture``).
"""
import argparse
import asyncio
import json

import browsesafe_pp as bs


def mock_transport():
    """A fake endpoint that answers 1 when the page contains an obvious 'Assistant note'."""
    import httpx

    def handler(request):
        page = json.loads(request.content)["messages"][1]["content"]
        verdict = int("assistant note" in str(page).lower())
        arguments = json.dumps({"classification": verdict, "explanation": "mock decision"})
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {
                "name": "record_classification", "arguments": arguments}}]}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}})

    return httpx.MockTransport(handler)


async def main(args) -> None:
    benchmark = bs.load(source=args.source, split=args.split, view=args.view, n_pairs=args.n_pairs)
    if args.mock:
        provider = bs.OpenAICompatible("http://mock.invalid/v1", "mock-model", transport=mock_transport())
    else:
        provider = bs.OpenAICompatible(model=args.model)
    detector = bs.ModelDetector(provider, max_tokens=args.max_tokens)
    try:
        report = await bs.evaluate_async(benchmark, detector, concurrency=args.concurrency)
    finally:
        await provider.aclose()
    report.manifest["detector"] = detector.configuration
    if args.output:
        report.save(args.output)
    print(json.dumps(report.metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", help="Local shard directory; omit to download from the Hub")
    parser.add_argument("--model", help="Model name (default: $BROWSESAFE_MODEL)")
    parser.add_argument("--mock", action="store_true", help="Use an offline stand-in model")
    parser.add_argument("--split", default="test")
    parser.add_argument("--view", default="html", choices=("html", "markdown", "accessibility", "screenshot"))
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
