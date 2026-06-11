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
from typing import Dict, List, Tuple


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

        batch: List[Tuple[str, str, int]] = []
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
                categories = ",".join(obj.get("categories", []))
                batch.append((arxiv_id, categories, offset))
                count += 1
                if len(batch) >= 50_000:
                    conn.executemany(
                        "INSERT INTO paper_meta (arxiv_id, categories, offset) "
                        "VALUES (?, ?, ?)",
                        batch,
                    )
                    batch.clear()

        if batch:
            conn.executemany(
                "INSERT INTO paper_meta (arxiv_id, categories, offset) VALUES (?, ?, ?)",
                batch,
            )
        conn.commit()
    finally:
        conn.close()

    if db_file.exists():
        db_file.unlink()
    Path(tmp_path).rename(db_file)
    return count


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
