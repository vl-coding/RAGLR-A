from src.rag_lit.preprocessing import (
    filter_by_candidate_ids,
    reduction_percent,
)
from src.rag_lit.schemas import Paper


def make_paper(arxiv_id: str, categories: list) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="Abstract.",
        year=2023,
        categories=categories,
    )


PAPERS = [
    make_paper("p1", ["cs.AI"]),
    make_paper("p2", ["cs.LG", "stat.ML"]),
    make_paper("p3", ["stat.TH"]),
    make_paper("p4", ["physics.optics"]),
    make_paper("p5", ["cs.CL"]),
]


def test_filter_by_candidate_ids_subset():
    result = filter_by_candidate_ids(PAPERS, {"p1", "p3"})
    ids = {p.arxiv_id for p in result}
    assert ids == {"p1", "p3"}


def test_filter_by_empty_candidate_ids_returns_all():
    result = filter_by_candidate_ids(PAPERS, set())
    assert len(result) == len(PAPERS)


def test_reduction_percent_basic():
    assert reduction_percent(1000, 100) == 90.0


def test_reduction_percent_zero_original():
    assert reduction_percent(0, 50) == 0.0


def test_reduction_percent_no_reduction():
    assert reduction_percent(500, 500) == 0.0
