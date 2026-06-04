import json
from pathlib import Path
from typing import Iterable, List
from datetime import datetime

from .schemas import Paper


def save_papers_jsonl(papers: List[Paper], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(paper.model_dump_json() + "\n")


def load_papers_jsonl(path: str) -> List[Paper]:
    papers = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(Paper.model_validate_json(line))

    return papers


def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    seen = {}
    for paper in papers:
        seen[paper.arxiv_id] = paper
    return list(seen.values())


def filter_papers_by_year_and_category(
    papers: List[Paper],
    min_year: int,
    allowed_categories: List[str],
) -> List[Paper]:
    filtered = []

    for paper in papers:
        has_category = any(cat in allowed_categories for cat in paper.categories)
        if paper.year >= min_year and has_category:
            filtered.append(paper)

    return filtered


def write_manifest(
    path: str,
    num_papers: int,
    categories: List[str],
    min_year: int,
    embedding_model: str,
) -> None:
    manifest = {
        "dataset_version": f"arxiv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "created_at_utc": datetime.utcnow().isoformat(),
        "num_papers": num_papers,
        "categories": categories,
        "min_year": min_year,
        "embedding_model": embedding_model,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)