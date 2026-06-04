import pytest

from src.rag_lit.preprocessing import (
    categories_for_selected_fields,
    filter_by_academic_fields,
    filter_by_candidate_ids,
    reduction_percent,
)
from src.rag_lit.schemas import Paper


MINI_CONFIG = {
    "academic_fields": {
        "all": {
            "label": "All arXiv Fields",
            "categories": "*",
        },
        "computer_science": {
            "label": "Computer Science",
            "categories": ["cs.AI", "cs.LG", "cs.CL"],
        },
        "statistics": {
            "label": "Statistics",
            "categories": ["stat.ML", "stat.TH"],
        },
    }
}


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


def test_categories_for_all_field_returns_none():
    result = categories_for_selected_fields(MINI_CONFIG, ["all"])
    assert result is None


def test_categories_for_empty_selection_returns_none():
    result = categories_for_selected_fields(MINI_CONFIG, [])
    assert result is None


def test_categories_for_specific_field():
    result = categories_for_selected_fields(MINI_CONFIG, ["computer_science"])
    assert set(result) == {"cs.AI", "cs.LG", "cs.CL"}


def test_categories_for_multiple_fields():
    result = categories_for_selected_fields(MINI_CONFIG, ["computer_science", "statistics"])
    assert "cs.AI" in result
    assert "stat.ML" in result


def test_unknown_field_raises():
    with pytest.raises(ValueError, match="Unknown academic field"):
        categories_for_selected_fields(MINI_CONFIG, ["unknown_field"])


def test_filter_by_all_returns_all_papers():
    result = filter_by_academic_fields(PAPERS, ["all"], MINI_CONFIG)
    assert len(result) == len(PAPERS)


def test_filter_by_cs_only():
    result = filter_by_academic_fields(PAPERS, ["computer_science"], MINI_CONFIG)
    ids = {p.arxiv_id for p in result}
    assert ids == {"p1", "p2", "p5"}


def test_filter_by_statistics_only():
    result = filter_by_academic_fields(PAPERS, ["statistics"], MINI_CONFIG)
    ids = {p.arxiv_id for p in result}
    assert ids == {"p2", "p3"}


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
