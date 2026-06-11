import re
import sqlite3
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


def save_keyword_index_db(index: Dict[str, Set[str]], db_path: str) -> None:
    """Write a token -> arxiv_ids inverted index to SQLite, atomically."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = str(db_file) + ".tmp"
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute(
            "CREATE TABLE postings (token TEXT PRIMARY KEY, arxiv_ids TEXT NOT NULL)"
        )
        batch = []
        for token, ids in index.items():
            batch.append((token, ",".join(ids)))
            if len(batch) >= 50_000:
                conn.executemany(
                    "INSERT INTO postings (token, arxiv_ids) VALUES (?, ?)", batch
                )
                batch.clear()
        if batch:
            conn.executemany(
                "INSERT INTO postings (token, arxiv_ids) VALUES (?, ?)", batch
            )
        conn.commit()
    finally:
        conn.close()

    if db_file.exists():
        db_file.unlink()
    Path(tmp_path).rename(db_file)


def open_keyword_index_db(db_path: str) -> sqlite3.Connection:
    """Open a read-only connection to the keyword postings DB.

    Lookups are point queries on the `token` primary key, so the OS page
    cache keeps this fast without ever loading the full index into the
    Python process.
    """
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def merge_new_papers_into_index_db(
    db_path: str,
    new_paper_tokens: List[Tuple[str, List[str]]],
) -> None:
    """Merge (arxiv_id, tokens) pairs into the postings DB in place."""
    conn = sqlite3.connect(db_path)
    try:
        for arxiv_id, tokens in new_paper_tokens:
            for token in set(tokens):
                row = conn.execute(
                    "SELECT arxiv_ids FROM postings WHERE token = ?", (token,)
                ).fetchone()
                if row:
                    ids = set(row[0].split(","))
                    if arxiv_id not in ids:
                        ids.add(arxiv_id)
                        conn.execute(
                            "UPDATE postings SET arxiv_ids = ? WHERE token = ?",
                            (",".join(ids), token),
                        )
                else:
                    conn.execute(
                        "INSERT INTO postings (token, arxiv_ids) VALUES (?, ?)",
                        (token, arxiv_id),
                    )
        conn.commit()
    finally:
        conn.close()


def candidate_ids_from_keywords(
    keywords: List[str],
    conn: sqlite3.Connection,
    mode: str = "union",
) -> Set[str]:
    matched_sets = []

    for keyword in keywords:
        keyword_tokens = tokenize(keyword)
        keyword_matches: Set[str] = set()

        for token in keyword_tokens:
            row = conn.execute(
                "SELECT arxiv_ids FROM postings WHERE token = ?", (token,)
            ).fetchone()
            if row:
                keyword_matches |= set(row[0].split(","))

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
