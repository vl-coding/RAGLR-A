import re
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .schemas import Paper


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text.lower())


def build_keyword_inverted_index(papers: List[Paper]) -> Dict[str, Set[str]]:
    index = defaultdict(set)

    for paper in papers:
        tokens = tokenize(paper.text)

        for token in set(tokens):
            index[token].add(paper.arxiv_id)

    return dict(index)


def save_keyword_index(index: Dict[str, Set[str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_keyword_index(path: str) -> Dict[str, Set[str]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def merge_new_papers_into_index(
    index: Dict[str, Set[str]],
    new_paper_tokens: List[Tuple[str, List[str]]],
) -> None:
    """Merge (arxiv_id, tokens) pairs into an existing index in-place."""
    for arxiv_id, tokens in new_paper_tokens:
        for token in set(tokens):
            if token in index:
                index[token].add(arxiv_id)
            else:
                index[token] = {arxiv_id}


def candidate_ids_from_keywords(
    keywords: List[str],
    keyword_index: Dict[str, Set[str]],
    mode: str = "union",
) -> Set[str]:
    matched_sets = []

    for keyword in keywords:
        keyword_tokens = tokenize(keyword)
        keyword_matches = set()

        for token in keyword_tokens:
            keyword_matches |= keyword_index.get(token, set())

        if keyword_matches:
            matched_sets.append(keyword_matches)

    if not matched_sets:
        return set()

    if mode == "intersection":
        result = matched_sets[0]
        for s in matched_sets[1:]:
            result = result & s
        return result

    result = set()
    for s in matched_sets:
        result |= s

    return result