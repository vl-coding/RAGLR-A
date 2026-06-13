from src.rag_lit.preprocessing import (
    arxiv_id_sort_key,
    candidate_ids_matching_categories,
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


def test_arxiv_id_sort_key_new_format_chronological():
    assert arxiv_id_sort_key("0704.0001") < arxiv_id_sort_key("1706.03762")
    assert arxiv_id_sort_key("1706.03762") < arxiv_id_sort_key("2606.07517")


def test_arxiv_id_sort_key_old_format_always_oldest():
    # Old-format ids (pre-2007) must sort before *every* new-format id,
    # even though "supr-con/..." > "1706.03762" as plain strings.
    assert arxiv_id_sort_key("quant-ph/9705052") < arxiv_id_sort_key("0704.0001")
    assert arxiv_id_sort_key("supr-con/9609004") < arxiv_id_sort_key("1706.03762")


def test_arxiv_id_sort_key_strips_version_suffix():
    assert arxiv_id_sort_key("2010.07626v1") == arxiv_id_sort_key("2010.07626")


def test_arxiv_id_sort_key_old_format_year_rollover():
    # YY >= 91 -> 19YY, YY < 91 -> 20YY (arxiv old-format ids run 1991-2007)
    assert arxiv_id_sort_key("hep-th/9711200")[0] == 1997
    assert arxiv_id_sort_key("math/0211159")[0] == 2002


def test_candidate_ids_matching_categories_single():
    result = candidate_ids_matching_categories(PAPERS, ["stat.TH"])
    assert result == {"p3"}


def test_candidate_ids_matching_categories_cross_listed():
    result = candidate_ids_matching_categories(PAPERS, ["stat.ML"])
    assert result == {"p2"}


def test_candidate_ids_matching_categories_union_of_multiple():
    result = candidate_ids_matching_categories(PAPERS, ["cs.AI", "cs.CL"])
    assert result == {"p1", "p5"}


def test_candidate_ids_matching_categories_no_match():
    result = candidate_ids_matching_categories(PAPERS, ["q-bio.NC"])
    assert result == set()
