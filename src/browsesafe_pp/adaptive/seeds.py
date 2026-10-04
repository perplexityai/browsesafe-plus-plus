"""Select unchanged injection fragments as starting points for adaptive search."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Mapping

from ..contracts import Seed
from ..dataset import Benchmark


def from_benchmark(benchmark: Benchmark, *, n: int = 8,
                   match: Mapping[str, Any] | None = None, query: str = "",
                   split: str = "train", seed: int = 42) -> tuple[Seed, ...]:
    """Pick ``n`` real injection fragments from ``benchmark`` as campaign seeds.

    Fragments are ranked by how well their pair matches the destination context and then
    diversified across (attack type, placement) cells:

    * ``match``: metadata preferences such as ``{"page_type": "article"}``; each matching
      key raises the score (preferences, not filters; use ``load(filters=...)`` to filter).
    * ``query``: free text, typically the destination task and attacker goal; word overlap
      with the source pair's task, goal and page type raises the score.

    Each seed is one byte-identical annotated fragment that occurs in the loaded view.
    Only pairs in ``split`` (default ``"train"``) are used, so test pairs stay unseen.
    """
    if type(n) is not int or n < 1 or type(seed) is not int:
        raise ValueError("n must be positive and seed must be an integer")
    revision = (benchmark.manifest.get("revision") or benchmark.manifest.get("fingerprint")
                or benchmark.manifest.get("sampling_identity"))
    if not revision:
        raise ValueError("Seed selection requires a source revision or fingerprint")
    split_column = benchmark.manifest.get("split_column", "split")
    terms = set(re.findall(r"\w+", query.lower()))
    unique = {}
    for pair in benchmark.pairs:
        page = pair.malicious
        row = page.metadata
        if row.get(split_column) != split:
            continue
        raw = row.get("injection_text_fragments_json")
        try:
            fragments = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            continue
        if not isinstance(fragments, list):
            continue
        score = 1.0
        for key, expected in (match or {}).items():
            actual = row.get(key)
            expected_values = expected if isinstance(expected, (list, tuple, set)) else [expected]
            actual_values = actual if isinstance(actual, (list, tuple, set)) else [actual]
            if any(value is not None and value in actual_values for value in expected_values):
                score += 2.0
        source_terms = set(re.findall(r"\w+", " ".join((
            str(row.get("benign_task") or ""), str(row.get("malicious_goal") or ""),
            str(row.get("page_type") or row.get("page_kind") or ""),
        )).lower()))
        if terms:
            score += 2 * len(terms & source_terms) / len(terms)
        cell = (str(row.get("attack_type") or "unknown"),
                str(row.get("injection_strategy") or "unknown"))
        for index, fragment in enumerate(fragments):
            if not isinstance(fragment, str) or not fragment.strip() or fragment not in page.content:
                continue
            identity = f"{page.entry_id}:fragment-{index}"
            tie = hashlib.sha256(f"{revision}\0{seed}\0{identity}".encode()).hexdigest()
            candidate = Seed(identity, fragment, revision, page.entry_id,
                             row.get("attack_identified_ts"))
            prior = unique.get(fragment)
            if prior is None or (score, tie) > prior[:2]:
                unique[fragment] = (score, tie, cell, candidate)
    pool = list(unique.values())
    if not pool:
        raise ValueError("No injection fragments found in the requested split of this benchmark")
    selected: list[Seed] = []
    cell_counts: Counter = Counter()
    while pool and len(selected) < n:
        best = max(range(len(pool)), key=lambda i: (
            pool[i][0] / (1 + cell_counts[pool[i][2]]), pool[i][1]))
        _, _, cell, candidate = pool.pop(best)
        selected.append(candidate)
        cell_counts[cell] += 1
    return tuple(selected)
