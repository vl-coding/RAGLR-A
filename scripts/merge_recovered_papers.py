"""
merge_recovered_papers.py

Merges arxiv_recovered.jsonl into arxiv_papers.jsonl, deduplicates by
arxiv_id, re-sorts, and writes the combined file.

Run this AFTER build_dense_index_fast.py finishes and BEFORE re-running
build_dense_index_fast.py (to index the recovered papers) and the BM25/keyword
index builds. If no recovery file is present, this step is a no-op.

Usage:
    python scripts/merge_recovered_papers.py
"""

import json
import time
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_yaml("configs/config.yaml")
    main_path = config["data"]["processed_path"]
    recovery_path = str(Path(main_path).parent / "arxiv_recovered.jsonl")

    if not Path(recovery_path).exists():
        print(f"Recovery file not found: {recovery_path} (nothing to merge, skipping)")
        return

    print(f"Loading main JSONL: {main_path}", flush=True)
    papers: dict = {}
    with open(main_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                aid = obj.get("arxiv_id")
                if aid:
                    papers[aid] = obj
            except json.JSONDecodeError:
                continue
            if (i + 1) % 500_000 == 0:
                print(f"  ... {i + 1:,} lines loaded", flush=True)
    print(f"Main JSONL: {len(papers):,} papers", flush=True)

    print(f"Loading recovery JSONL: {recovery_path}", flush=True)
    before = len(papers)
    with open(recovery_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                aid = obj.get("arxiv_id")
                if aid:
                    papers[aid] = obj
            except json.JSONDecodeError:
                continue
    added = len(papers) - before
    print(f"Recovery JSONL added: {added:,} new papers", flush=True)
    print(f"Combined total: {len(papers):,} papers", flush=True)

    print(f"Writing merged JSONL (sorted by arxiv_id): {main_path}", flush=True)
    start = time.time()
    with open(main_path, "w", encoding="utf-8") as f:
        for arxiv_id in sorted(papers.keys()):
            f.write(json.dumps(papers[arxiv_id], ensure_ascii=False) + "\n")
    print(f"Done in {(time.time() - start) / 60:.1f}m", flush=True)
    print(flush=True)
    print("Recovery file can now be deleted:", flush=True)
    print(f"  del {recovery_path}", flush=True)


if __name__ == "__main__":
    main()
