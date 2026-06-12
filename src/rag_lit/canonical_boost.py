from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import yaml


def load_canonical_papers(path: str) -> List[Dict[str, Any]]:
    """Load the canonical-papers registry (arxiv_id, title, topics) from YAML.

    Returns an empty list if `path` is falsy or the file doesn't exist, so
    this feature can be disabled by leaving `retrieval.canonical_papers_path`
    unset in config.yaml.
    """
    if not path or not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def match_canonical_papers(
    query: str,
    keywords: Sequence[str],
    canonical_papers: List[Dict[str, Any]],
    corpus_ids: Set[str],
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Rank canonical papers whose `topics` match the query, RRF-list format.

    A canonical paper is included if at least one of its `topics` phrases
    appears as a substring of the lowercased query or the lowercased,
    space-joined keyword list. Papers are ranked by the total character
    length of their matched topic phrases, so longer/more specific phrase
    matches (e.g. "parameter-efficient fine-tuning") outrank short, generic
    ones (e.g. "large language model") that happen to also appear in the
    query -- higher rank => larger RRF contribution when this list is fused
    with dense/BM25 results.
    """
    if not canonical_papers:
        return []

    query_lower = query.lower()
    keyword_text = " ".join(keywords).lower()

    scored = []
    for paper in canonical_papers:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id or arxiv_id not in corpus_ids:
            continue

        match_score = sum(
            len(topic)
            for topic in paper.get("topics", [])
            if topic.lower() in query_lower or topic.lower() in keyword_text
        )
        if match_score > 0:
            scored.append((match_score, arxiv_id))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {"doc_id": arxiv_id, "rank": i + 1, "source": "canonical"}
        for i, (_, arxiv_id) in enumerate(scored[:max_results])
    ]
