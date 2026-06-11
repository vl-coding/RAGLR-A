"""
build_metadata_db.py

One-time (offline) build of the SQLite metadata index
(arxiv_id -> categories, byte offset) over the main processed JSONL corpus.

Pipeline startup loads this DB with a single bulk SELECT instead of running
json.loads over every line of a multi-GB JSONL file (which previously took
30+ minutes for 3M+ papers).

Re-run this script whenever data/processed/arxiv_papers.jsonl changes
(e.g. after merge_recovered_papers.py). The pipeline will also rebuild the
DB automatically if it's missing or older than the JSONL, but running this
ahead of time avoids paying that cost on a live session.

Usage:
    python scripts/build_metadata_db.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.metadata_db import build_metadata_db


def main() -> None:
    t0 = time.time()
    config = load_config()
    ensure_project_dirs(config)

    jsonl_path = config["data"]["processed_path"]
    db_path = config["paths"]["metadata_db"]

    print(f"Building metadata DB from {jsonl_path} -> {db_path} ...", flush=True)
    count = build_metadata_db(jsonl_path, db_path)
    print(f"Done. {count:,} papers indexed in {time.time() - t0:.1f}s -> {db_path}", flush=True)


if __name__ == "__main__":
    main()
