import math
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import yaml


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for doc_id in relevant_ids if doc_id in top_k)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains[:k]))


def ndcg_at_k(retrieved_ids: Sequence[str], relevance: Dict[str, float], k: int) -> float:
    """relevance maps doc_id -> graded relevance score; docs missing from it score 0."""
    gains = [relevance.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]
    dcg = dcg_at_k(gains, k)

    ideal_gains = sorted(relevance.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def set_recall(candidate_ids: Set[str], relevant_ids: Set[str]) -> float:
    """Recall of an unordered candidate set against relevant_ids.

    Used to check whether the keyword prefilter stage drops known-relevant
    papers before they reach dense/BM25.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in relevant_ids if doc_id in candidate_ids)
    return hits / len(relevant_ids)


def rank_of(doc_id: str, ranked_ids: Sequence[str]) -> Optional[int]:
    """1-based rank of doc_id in ranked_ids, or None if it doesn't appear."""
    for i, candidate in enumerate(ranked_ids, start=1):
        if candidate == doc_id:
            return i
    return None


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, Optional[float]]:
    """Bootstrap CI on the mean of values via resampling with replacement.

    Returns {"mean", "low", "high"} (rounded to 4 dp), or all-None for an
    empty input.
    """
    if not values:
        return {"mean": None, "low": None, "high": None}

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()

    alpha = (1 - ci) / 2
    low_idx = int(alpha * n_resamples)
    high_idx = min(int((1 - alpha) * n_resamples), n_resamples - 1)

    return {
        "mean": round(statistics.mean(values), 4),
        "low": round(means[low_idx], 4),
        "high": round(means[high_idx], 4),
    }


def load_gold_queries(path: str = "tests/eval/gold_queries.yaml") -> List[dict]:
    """Load the gold query set: list of {query, relevant_ids}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or []
