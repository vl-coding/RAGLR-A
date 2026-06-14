"""
SQLite-backed metadata index: arxiv_id -> (categories, byte offset into the
processed JSONL).

Built once (offline, via scripts/build_metadata_db.py, or lazily on first
use) so that pipeline startup loads metadata with a single bulk SELECT
instead of running json.loads over every line of a multi-GB JSONL file.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def build_metadata_db(jsonl_path: str, db_path: str) -> int:
    """Build the metadata DB from jsonl_path, writing atomically to db_path.

    Returns the number of papers indexed.
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = str(db_file) + ".tmp"
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute(
            "CREATE TABLE paper_meta ("
            "arxiv_id TEXT PRIMARY KEY, "
            "categories TEXT NOT NULL, "
            "offset INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE paper_categories ("
            "arxiv_id TEXT NOT NULL, "
            "category TEXT NOT NULL)"
        )

        batch: List[Tuple[str, str, int]] = []
        category_batch: List[Tuple[str, str]] = []
        count = 0
        with open(jsonl_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                arxiv_id = obj.get("arxiv_id", "")
                paper_categories = obj.get("categories", [])
                categories = ",".join(paper_categories)
                batch.append((arxiv_id, categories, offset))
                category_batch.extend((arxiv_id, category) for category in paper_categories)
                count += 1
                if len(batch) >= 50_000:
                    conn.executemany(
                        "INSERT INTO paper_meta (arxiv_id, categories, offset) "
                        "VALUES (?, ?, ?)",
                        batch,
                    )
                    batch.clear()
                if len(category_batch) >= 50_000:
                    conn.executemany(
                        "INSERT INTO paper_categories (arxiv_id, category) VALUES (?, ?)",
                        category_batch,
                    )
                    category_batch.clear()

        if batch:
            conn.executemany(
                "INSERT INTO paper_meta (arxiv_id, categories, offset) VALUES (?, ?, ?)",
                batch,
            )
        if category_batch:
            conn.executemany(
                "INSERT INTO paper_categories (arxiv_id, category) VALUES (?, ?)",
                category_batch,
            )
        conn.execute(
            "CREATE INDEX idx_paper_categories_category ON paper_categories (category)"
        )
        conn.commit()
    finally:
        conn.close()

    if db_file.exists():
        db_file.unlink()
    Path(tmp_path).rename(db_file)
    return count


def metadata_db_has_category_table(db_path: str) -> bool:
    """True if db_path has the paper_categories table (issue #5 phase 2 schema).

    Older metadata DBs predate this table; pipeline startup uses this to
    force a rebuild even if the DB is otherwise newer than the source JSONL.
    """
    if not Path(db_path).exists():
        return False

    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_categories'"
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def load_metadata_db(db_path: str) -> Tuple[List[Tuple[str, List[str]]], Dict[str, int]]:
    """Load (arxiv_id, categories) rows and the arxiv_id -> offset map."""
    rows: List[Tuple[str, List[str]]] = []
    offsets: Dict[str, int] = {}

    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        for arxiv_id, categories, offset in conn.execute(
            "SELECT arxiv_id, categories, offset FROM paper_meta"
        ):
            offsets[arxiv_id] = offset
            rows.append((arxiv_id, categories.split(",") if categories else []))
    finally:
        conn.close()

    return rows, offsets


def open_metadata_db(db_path: str) -> sqlite3.Connection:
    """Open a read-only connection to the metadata DB for query-time lookups.

    Used to filter by category via the indexed `paper_categories` table
    without materializing/iterating the full in-memory paper list per query.
    """
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def candidate_ids_for_categories(conn: sqlite3.Connection, categories: Iterable[str]) -> Set[str]:
    """Returns arxiv_ids of papers in the main corpus whose categories overlap `categories`.

    Cross-listing aware (a paper matches if *any* of its categories is in
    the requested set), backed by the indexed `paper_categories` table so
    this is an indexed lookup rather than a full scan of the corpus.
    """
    categories = list(categories)
    if not categories:
        return set()

    placeholders = ",".join("?" * len(categories))
    rows = conn.execute(
        f"SELECT DISTINCT arxiv_id FROM paper_categories WHERE category IN ({placeholders})",
        categories,
    ).fetchall()
    return {row[0] for row in rows}
