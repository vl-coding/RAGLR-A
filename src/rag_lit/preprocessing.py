from typing import List, Set

from .schemas import Paper


def filter_by_candidate_ids(papers: List[Paper], candidate_ids: Set[str]) -> List[Paper]:
    if not candidate_ids:
        return papers

    return [paper for paper in papers if paper.arxiv_id in candidate_ids]


def apply_candidate_safety_rules(
    original_papers: List[Paper],
    filtered_papers: List[Paper],
    min_candidates: int,
    max_candidates: int,
) -> List[Paper]:
    if len(filtered_papers) < min_candidates:
        return original_papers

    if len(filtered_papers) > max_candidates:
        return filtered_papers[:max_candidates]

    return filtered_papers


def reduction_percent(original_count: int, new_count: int) -> float:
    if original_count == 0:
        return 0.0

    return round((1 - new_count / original_count) * 100, 2)
