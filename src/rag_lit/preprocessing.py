import re
from typing import List, Set, Tuple

from .schemas import Paper

_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
_NEW_ID_RE = re.compile(r"^(\d{2})(\d{2})\.(\d+)$")
_OLD_ID_RE = re.compile(r"^[a-zA-Z][\w\-.]*/(\d{2})(\d{2})(\d+)$")


def arxiv_id_sort_key(arxiv_id: str) -> Tuple[int, int, int]:
    """Chronological sort key for an arxiv id, as (year, month, sequence).

    Handles both the post-2007 'YYMM.NNNNN' format and the pre-2007
    'category/YYMMNNN' format (e.g. "quant-ph/9705052"). A naive descending
    string sort puts every old-format id (starting with a letter) ahead of
    every new-format id (starting with a digit), so selecting "the most
    recent N papers" via `sorted(ids, reverse=True)[:n]` silently drops
    ~17 years of new-format papers in favor of all pre-2007 ones.
    """
    clean = _VERSION_SUFFIX_RE.sub("", arxiv_id)

    m = _NEW_ID_RE.match(clean)
    if m:
        yy, mm, seq = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (2000 + yy, mm, seq)

    m = _OLD_ID_RE.match(clean)
    if m:
        yy, mm, seq = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 1900 + yy if yy >= 91 else 2000 + yy
        return (year, mm, seq)

    return (0, 0, 0)


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
