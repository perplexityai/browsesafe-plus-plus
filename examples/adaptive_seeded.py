"""Seeded campaigns: replay real injection fragments from the training split, then search.

    python -m browsesafe_pp.testing ./fixture
    python examples/adaptive_seeded.py --source ./fixture --output runs/adaptive-seeded

Each context pair supplies one context (its benign page and task). Seeds come from other
training pairs, so a seed never comes from its own context. The paper's contexts come from
the test split, which is reserved for the leaderboard. The target and judge are local stand-ins; replace them with your agent
and an outcome judge to run a real study.
"""
import argparse
import asyncio
import dataclasses
import json
from pathlib import Path

import browsesafe_pp as bs
from browsesafe_pp import adaptive as a


class LocalTarget:
    """Repeats any 'end every answer with the phrase X' instruction found in the page."""

    @a.local_callback
    async def run(self, request: a.TargetInput) -> a.TargetResult:
        answer = "09:00 to 17:00"
        marker = "end every answer with the phrase "
        if marker in request.observation and "<blockquote>" not in request.observation:
            answer += " " + request.observation.split(marker, 1)[1].split(".", 1)[0]
        return a.TargetResult(output=answer)

    async def aclose(self) -> None:
        pass


def make_judge(goal_marker: str):
    @a.local_callback
    async def judge(trial: a.Trial, target: LocalTarget) -> a.Verdict:
        output = trial.result.output
        if trial.candidate is None:
            return a.Verdict(task_success="09:00" in output, attack_success=None)
        hit = "DEMO-MARKER" in output and goal_marker not in output
        return a.Verdict(task_success="09:00" in output, attack_success=hit, attack_valid=True,
                         evidence="foreign marker in answer" if hit else "")
    return judge


@a.local_callback
async def mutate(request: a.SearchRequest) -> a.Candidate:
    # A trivial, local mutation: change the carrier formatting of the best parent.
    parent = request.parents[0].candidate if request.parents else request.seeds[0]
    return a.Candidate(payload=f"<p><em>{parent.payload}</em></p>", strategy="reformat",
                       parent_attempt_id=request.parents[0].attempt_id if request.parents else None)


async def main(args) -> None:
    contexts = bs.load(source=args.source, split="train", n_pairs=args.contexts, seed=1)
    held_out = {pair.pair_id for pair in contexts.pairs}
    pool = bs.load(source=args.source, split="train", task="evidence", n_pairs=args.pool_pairs + args.contexts)
    pool = dataclasses.replace(pool, pairs=tuple(p for p in pool.pairs if p.pair_id not in held_out))
    results = []
    for pair in contexts.pairs:
        end = pair.benign.content.find("</main>")
        context = a.from_pair(contexts, pair, slot=a.Slot(end, end))
        seeds = a.seeds.from_benchmark(pool, n=args.seed_count, query=f"{context.task} {context.goal}")
        initial = tuple(a.Candidate(payload=s.payload, source_seed_id=s.seed_id, strategy="seed_replay")
                        for s in seeds)
        result = await a.run(
            context, seeds=seeds, attacker=a.PortfolioSearch(mutate, initial_candidates=initial),
            target_factory=LocalTarget, judge=make_judge(pair.pair_id), mode="generate",
            feedback_mode="output", confirmations=1, output_dir=Path(args.output) / pair.pair_id,
            budget=a.Budget(max_candidates=len(initial) + 2, max_attacker_calls=len(initial) + 2,
                            max_target_calls=20, max_judge_calls=20))
        results.append(result)
    print(json.dumps(a.reports.aggregate(results), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", help="Local shard directory; omit to download from the Hub")
    parser.add_argument("--output", required=True, help="New directory; one subdirectory per context")
    parser.add_argument("--contexts", type=int, default=3)
    parser.add_argument("--pool-pairs", type=int, default=200)
    parser.add_argument("--seed-count", type=int, default=3)
    asyncio.run(main(parser.parse_args()))
