from src.rag_lit.eval_metrics import (
    bootstrap_ci,
    dcg_at_k,
    load_gold_queries,
    mrr,
    ndcg_at_k,
    precision_at_k,
    rank_of,
    recall_at_k,
    set_recall,
)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"B", "D", "Z"}
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=4) == 0.5
    assert precision_at_k([], relevant, k=4) == 0.0


def test_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"B", "D", "Z"}
    assert recall_at_k(retrieved, relevant, k=2) == 1 / 3
    assert recall_at_k(retrieved, relevant, k=4) == 2 / 3
    assert recall_at_k(retrieved, set(), k=4) == 0.0


def test_mrr():
    relevant = {"C"}
    assert mrr(["A", "B", "C", "D"], relevant) == 1 / 3
    assert mrr(["C", "A"], relevant) == 1.0
    assert mrr(["A", "B"], relevant) == 0.0


def test_dcg_at_k():
    assert dcg_at_k([], 5) == 0.0
    assert dcg_at_k([1.0], 5) == 1.0  # log2(2) == 1
    # Earlier positions contribute more for equal gains.
    assert dcg_at_k([1.0, 1.0], 1) < dcg_at_k([1.0, 1.0], 2)


def test_ndcg_at_k_perfect_ranking():
    relevance = {"A": 3.0, "B": 2.0, "C": 1.0}
    assert ndcg_at_k(["A", "B", "C"], relevance, k=3) == 1.0


def test_ndcg_at_k_worst_ranking():
    relevance = {"A": 3.0, "B": 2.0, "C": 1.0}
    perfect = ndcg_at_k(["A", "B", "C"], relevance, k=3)
    worst = ndcg_at_k(["C", "B", "A"], relevance, k=3)
    assert worst < perfect


def test_ndcg_at_k_empty_relevance():
    assert ndcg_at_k(["A", "B"], {}, k=2) == 0.0


def test_set_recall():
    relevant = {"A", "B", "C"}
    assert set_recall({"A", "B", "C", "D"}, relevant) == 1.0
    assert set_recall({"A"}, relevant) == 1 / 3
    assert set_recall(set(), relevant) == 0.0
    assert set_recall({"A"}, set()) == 0.0


def test_load_gold_queries():
    entries = load_gold_queries("tests/eval/gold_queries.yaml")
    assert len(entries) > 0
    for entry in entries:
        assert "query" in entry
        assert "relevant_ids" in entry
        assert isinstance(entry["relevant_ids"], list)
        assert entry.get("stratum") in ("terminology_aligned", "terminology_gap")


def test_rank_of():
    ranked = ["A", "B", "C"]
    assert rank_of("A", ranked) == 1
    assert rank_of("C", ranked) == 3
    assert rank_of("Z", ranked) is None
    assert rank_of("A", []) is None


def test_bootstrap_ci_empty():
    result = bootstrap_ci([])
    assert result == {"mean": None, "low": None, "high": None}


def test_bootstrap_ci_constant_values():
    result = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_resamples=200, seed=1)
    assert result["mean"] == 1.0
    assert result["low"] == 1.0
    assert result["high"] == 1.0


def test_bootstrap_ci_bounds_contain_mean():
    values = [0.0, 0.0, 1.0, 1.0, 1.0]
    result = bootstrap_ci(values, n_resamples=500, seed=7)
    assert result["low"] <= result["mean"] <= result["high"]
    assert 0.0 <= result["low"]
    assert result["high"] <= 1.0


def test_bootstrap_ci_deterministic_with_seed():
    values = [0.1, 0.5, 0.9, 0.3, 0.7]
    r1 = bootstrap_ci(values, n_resamples=300, seed=42)
    r2 = bootstrap_ci(values, n_resamples=300, seed=42)
    assert r1 == r2
