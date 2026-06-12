import pytest

from src.rag_lit.rrf import reciprocal_rank_fusion


def test_basic_fusion():
    dense = [
        {"doc_id": "A", "rank": 1, "score": 0.9, "source": "dense"},
        {"doc_id": "B", "rank": 2, "score": 0.8, "source": "dense"},
        {"doc_id": "C", "rank": 3, "score": 0.7, "source": "dense"},
    ]
    bm25 = [
        {"doc_id": "B", "rank": 1, "score": 15.0, "source": "bm25"},
        {"doc_id": "A", "rank": 2, "score": 12.0, "source": "bm25"},
        {"doc_id": "D", "rank": 3, "score": 9.0, "source": "bm25"},
    ]

    fused = reciprocal_rank_fusion([dense, bm25], k=60)

    doc_ids = [item["doc_id"] for item in fused]

    # B appears at rank 1 and 2 across systems so should rank highly
    assert "B" in doc_ids[:2]
    assert all("rrf_score" in item for item in fused)
    assert all("dense_rank" in item for item in fused)
    assert all("bm25_rank" in item for item in fused)


def test_ranks_are_consecutive():
    dense = [{"doc_id": "X", "rank": 1, "score": 1.0, "source": "dense"}]
    bm25 = [{"doc_id": "Y", "rank": 1, "score": 1.0, "source": "bm25"}]

    fused = reciprocal_rank_fusion([dense, bm25])

    ranks = [item["rank"] for item in fused]
    assert ranks == list(range(1, len(fused) + 1))


def test_empty_lists():
    fused = reciprocal_rank_fusion([[], []])
    assert fused == []


def test_single_list():
    single = [{"doc_id": "A", "rank": 1, "score": 5.0, "source": "dense"}]
    fused = reciprocal_rank_fusion([single])
    assert len(fused) == 1
    assert fused[0]["doc_id"] == "A"


def test_rrf_score_decreases_with_rank():
    dense = [
        {"doc_id": str(i), "rank": i, "score": float(10 - i), "source": "dense"}
        for i in range(1, 6)
    ]
    fused = reciprocal_rank_fusion([dense], k=60)

    scores = [item["rrf_score"] for item in fused]
    assert scores == sorted(scores, reverse=True)


def test_source_tracking():
    dense = [{"doc_id": "A", "rank": 1, "score": 1.0, "source": "dense"}]
    bm25 = [{"doc_id": "A", "rank": 2, "score": 1.0, "source": "bm25"}]

    fused = reciprocal_rank_fusion([dense, bm25])

    a = next(item for item in fused if item["doc_id"] == "A")
    assert a["dense_rank"] == 1
    assert a["bm25_rank"] == 2


def test_canonical_list_can_boost_a_doc_missing_from_other_retrievers():
    dense = [
        {"doc_id": "X", "rank": i, "score": 1.0, "source": "dense"}
        for i in range(1, 11)
    ]
    bm25 = [
        {"doc_id": "Y", "rank": i, "score": 1.0, "source": "bm25"}
        for i in range(1, 11)
    ]
    canonical = [{"doc_id": "Z", "rank": 1, "source": "canonical"}]

    fused = reciprocal_rank_fusion([dense, bm25, canonical], k=60)

    z = next(item for item in fused if item["doc_id"] == "Z")
    assert z["rrf_score"] == pytest.approx(1.0 / 61)
    assert z["dense_rank"] is None
    assert z["bm25_rank"] is None
