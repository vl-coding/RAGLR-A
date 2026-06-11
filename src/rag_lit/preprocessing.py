from typing import List, Optional, Set

from .schemas import Paper


def categories_for_selected_fields(
    config: dict,
    selected_fields: List[str],
) -> Optional[List[str]]:
    field_config = config["academic_fields"]

    if not selected_fields or "all" in selected_fields:
        return None  # None means no category filter

    categories: Set[str] = set()

    for field_key in selected_fields:
        if field_key not in field_config:
            raise ValueError(f"Unknown academic field: {field_key}")

        cats = field_config[field_key]["categories"]

        if cats == "*":
            return None  # Wildcard means all categories

        for cat in cats:
            categories.add(cat)

    return sorted(categories)


def filter_by_academic_fields(
    papers: List[Paper],
    selected_fields: List[str],
    config: dict,
    category_override: Optional[List[str]] = None,
) -> List[Paper]:
    if category_override is not None:
        allowed_categories = category_override
    else:
        allowed_categories = categories_for_selected_fields(config, selected_fields)

    if allowed_categories is None:
        return papers

    return [
        paper for paper in papers
        if any(cat in allowed_categories for cat in paper.categories)
    ]


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
