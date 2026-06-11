"""
incremental_update.py

Runs a fast incremental update of the arXiv corpus and all pre-query indexes.
Designed to be called twice a day (by run_scheduler.py) with no pipeline restart.

What it does:
  1. Harvests new arXiv papers via OAI-PMH since the last run
  2. Filters to papers not already in the corpus (using known_ids.npy)
  3. Appends new papers to data/processed/arxiv_delta.jsonl
  4. Embeds and upserts new papers into the ChromaDB dense index
  5. Rebuilds the delta BM25 index (covers all papers in arxiv_delta.jsonl)
  6. Merges new paper tokens into the keyword inverted index (atomic write)
  7. Extends known_ids.npy with the new IDs
  8. Updates update_state.json

The running pipeline (pipeline.py) detects index file changes via mtime and
hot-reloads them between queries — no restart or downtime required.

Usage:
    python scripts/incremental_update.py
    python scripts/incremental_update.py --dry-run
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import numpy as np

from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.schemas import Paper
from src.rag_lit.bm25_retriever import BM25Retriever
from src.rag_lit.keyword_index import merge_new_papers_into_index_db, tokenize
from src.rag_lit.dense_retriever import DenseRetriever
from update_arxiv_data import harvest_oai_records, load_state, save_state, utc_now_iso


# ---------------------------------------------------------------------------
# Known-IDs helpers (fast dedup without loading 3 GB JSONL)
# ---------------------------------------------------------------------------

def load_known_ids(known_ids_path: str, bm25_ids_path: str, delta_jsonl_path: str) -> set:
    """
    Load the set of all arxiv_ids already in the corpus.
    Bootstraps from BM25 arxiv_ids.npy on first run, then maintained incrementally.
    """
    path = Path(known_ids_path)
    if path.exists():
        return set(np.load(str(path), allow_pickle=True).tolist())

    print("known_ids.npy not found — bootstrapping from BM25 index and delta JSONL ...", flush=True)
    ids: set = set()

    bm25_npy = Path(bm25_ids_path) / "arxiv_ids.npy"
    if bm25_npy.exists():
        ids.update(np.load(str(bm25_npy), allow_pickle=True).tolist())
        print(f"  Loaded {len(ids):,} IDs from BM25 index", flush=True)

    delta = Path(delta_jsonl_path)
    if delta.exists():
        with open(str(delta), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        arxiv_id = json.loads(line).get("arxiv_id")
                        if arxiv_id:
                            ids.add(arxiv_id)
                    except json.JSONDecodeError:
                        pass
        print(f"  Total after delta: {len(ids):,} IDs", flush=True)

    np.save(str(path), np.array(sorted(ids), dtype=object))
    print(f"  Saved to {path}", flush=True)
    return ids


def save_known_ids(known_ids_path: str, known_ids: set) -> None:
    np.save(known_ids_path, np.array(sorted(known_ids), dtype=object))


# ---------------------------------------------------------------------------
# New-paper helpers
# ---------------------------------------------------------------------------

def papers_to_jsonl_lines(papers: list) -> list:
    """Convert raw OAI dicts to Paper-validated JSON strings."""
    lines = []
    for p in papers:
        try:
            paper = Paper(
                arxiv_id=p["arxiv_id"],
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                authors=p.get("authors", []),
                categories=p.get("categories", []),
                year=p.get("year", 0),
                url=p.get("url"),
                published_date=p.get("published_date") or p.get("updated_at_utc"),
                updated_date=p.get("updated_date") or p.get("updated_at_utc"),
            )
            lines.append(paper.model_dump_json())
        except Exception as e:
            print(f"  Skipping malformed paper {p.get('arxiv_id')}: {e}", flush=True)
    return lines


def append_to_delta_jsonl(delta_path: str, jsonl_lines: list) -> None:
    Path(delta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(delta_path, "a", encoding="utf-8") as f:
        for line in jsonl_lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Index update helpers
# ---------------------------------------------------------------------------

def embed_new_papers(config: dict, new_papers: list) -> None:
    """Embed and upsert new papers into ChromaDB (idempotent by arxiv_id)."""
    dense = DenseRetriever(
        model_name=config["models"]["embedding_model"],
        persist_dir=config["paths"]["dense_index_dir"],
    )
    paper_objects = []
    for p in new_papers:
        try:
            paper_objects.append(Paper(
                arxiv_id=p["arxiv_id"],
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                authors=p.get("authors", []),
                categories=p.get("categories", []),
                year=p.get("year", 0),
                url=p.get("url"),
            ))
        except Exception:
            pass
    dense.build_index(paper_objects)
    print(f"  Upserted {len(paper_objects):,} papers into ChromaDB", flush=True)


def rebuild_delta_bm25(delta_jsonl_path: str, bm25_delta_path: str) -> None:
    """Rebuild the delta BM25 index over all papers in arxiv_delta.jsonl."""
    tmp_path = bm25_delta_path + "_tmp"
    if Path(tmp_path).exists():
        shutil.rmtree(tmp_path)

    arxiv_ids = []
    corpus_tokens = []
    with open(delta_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                arxiv_id = obj.get("arxiv_id", "")
                text = f"{obj.get('title', '')}\n\n{obj.get('abstract', '')}"
                arxiv_ids.append(arxiv_id)
                corpus_tokens.append(tokenize(text))
            except json.JSONDecodeError:
                pass

    import bm25s
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)

    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    bm25.save(str(Path(tmp_path) / "index"))
    np.save(str(Path(tmp_path) / "arxiv_ids.npy"), np.array(arxiv_ids, dtype=object))

    # Atomic swap: remove old, rename new
    old_path = bm25_delta_path + "_old"
    if Path(old_path).exists():
        shutil.rmtree(old_path)
    if Path(bm25_delta_path).exists():
        os.rename(bm25_delta_path, old_path)
    os.rename(tmp_path, bm25_delta_path)
    if Path(old_path).exists():
        shutil.rmtree(old_path)

    print(f"  Delta BM25 rebuilt: {len(arxiv_ids):,} papers -> {bm25_delta_path}", flush=True)


def merge_keyword_index(keyword_index_path: str, new_papers: list) -> None:
    """Merge new paper tokens directly into the keyword postings DB."""
    new_paper_tokens = []
    for p in new_papers:
        text = f"{p.get('title', '')}\n\n{p.get('abstract', '')}"
        new_paper_tokens.append((p["arxiv_id"], tokenize(text)))

    merge_new_papers_into_index_db(keyword_index_path, new_paper_tokens)
    print(f"  Keyword index updated: {len(new_paper_tokens):,} papers merged", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Harvest and print count of new papers without writing anything")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    t0 = time.time()
    config = load_config(args.config)
    ensure_project_dirs(config)

    processed_path = config["data"]["processed_path"]
    delta_path = config["data"]["delta_path"]
    bm25_index_path = config["paths"]["bm25_index"]
    bm25_delta_path = config["paths"]["bm25_delta"]
    keyword_index_path = config["paths"]["keyword_index"]
    known_ids_path = config["paths"]["known_ids"]
    state_path = config["paths"]["update_state"]
    base_url = config["data"]["oai_base_url"]
    metadata_prefix = config["data"].get("oai_metadata_prefix", "oai_dc")

    # Load state
    state = load_state(state_path)
    last_harvest_date = state.get("last_successful_harvest_date")

    print(f"[{utc_now_iso()}] Incremental update started", flush=True)
    print(f"  Last harvest: {last_harvest_date or 'never'}", flush=True)

    # Load known IDs for fast dedup
    known_ids = load_known_ids(known_ids_path, bm25_index_path, delta_path)
    print(f"  Known IDs: {len(known_ids):,}", flush=True)

    # Harvest from OAI-PMH
    print("Harvesting from OAI-PMH ...", flush=True)
    harvested = harvest_oai_records(
        base_url=base_url,
        metadata_prefix=metadata_prefix,
        from_date=last_harvest_date,
        sleep_seconds=3.0,
    )
    print(f"Harvested {len(harvested):,} records from OAI-PMH", flush=True)

    # Filter to truly new papers
    new_papers = [p for p in harvested if p.get("arxiv_id") and p["arxiv_id"] not in known_ids]
    print(f"New papers not yet in corpus: {len(new_papers):,}", flush=True)

    if not new_papers:
        print("Nothing to do.", flush=True)
        # Still update state so next run uses today as from_date
        state.update({
            "last_successful_harvest_at_utc": utc_now_iso(),
            "last_successful_harvest_date": datetime.now(timezone.utc).date().isoformat(),
            "last_run_new_papers": 0,
        })
        save_state(state_path, state)
        return

    if args.dry_run:
        print(f"[dry-run] Would add {len(new_papers):,} papers. Exiting.", flush=True)
        return

    # 1. Append to delta JSONL (source of truth for delta papers)
    print(f"Appending {len(new_papers):,} papers to {delta_path} ...", flush=True)
    jsonl_lines = papers_to_jsonl_lines(new_papers)
    if jsonl_lines:
        append_to_delta_jsonl(delta_path, jsonl_lines)
    actual_new = len(jsonl_lines)
    print(f"  {actual_new:,} papers appended", flush=True)

    if actual_new == 0:
        print("No valid papers after schema validation. Exiting.", flush=True)
        return

    # 2. Embed and upsert into ChromaDB
    print("Embedding new papers -> ChromaDB ...", flush=True)
    embed_new_papers(config, new_papers[:actual_new])

    # 3. Rebuild delta BM25 (covers all papers in arxiv_delta.jsonl)
    print(f"Rebuilding delta BM25 index from {delta_path} ...", flush=True)
    rebuild_delta_bm25(delta_path, bm25_delta_path)

    # 4. Merge new papers into keyword index
    print("Merging new papers into keyword index ...", flush=True)
    merge_keyword_index(keyword_index_path, new_papers[:actual_new])

    # 5. Update known_ids
    new_ids = {p["arxiv_id"] for p in new_papers[:actual_new]}
    known_ids.update(new_ids)
    save_known_ids(known_ids_path, known_ids)
    print(f"  known_ids updated: {len(known_ids):,} total", flush=True)

    # 6. Update state
    state.update({
        "last_successful_harvest_at_utc": utc_now_iso(),
        "last_successful_harvest_date": datetime.now(timezone.utc).date().isoformat(),
        "total_known_papers": len(known_ids),
        "last_run_new_papers": actual_new,
    })
    save_state(state_path, state)

    elapsed = time.time() - t0
    print(f"[{utc_now_iso()}] Done. {actual_new:,} new papers indexed in {elapsed/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
