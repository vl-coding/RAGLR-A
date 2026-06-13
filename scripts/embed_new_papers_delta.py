"""
embed_new_papers_delta.py

After a full re-harvest (scripts/update_arxiv_data.py reset mode), the
existing ChromaDB dense index already has embeddings for the previous
corpus. Re-embedding all 3M+ papers to pick up a small number of newly
harvested papers would take ~24-28h for no benefit, since Paper.text
(title+abstract) for pre-existing papers is unchanged.

This script finds papers in data/processed/arxiv_papers.jsonl whose
arxiv_id is not yet in the ChromaDB collection, and embeds + upserts only
those.

Usage:
    python scripts/embed_new_papers_delta.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb

from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.schemas import Paper
from src.rag_lit.dense_retriever import DenseRetriever


def main() -> None:
    t0 = time.time()
    config = load_config()
    ensure_project_dirs(config)

    jsonl_path = config["data"]["processed_path"]
    persist_dir = config["paths"]["dense_index_dir"]

    print(f"Loading existing IDs from ChromaDB ({persist_dir}) ...", flush=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("arxiv_papers")
    existing_ids = set(collection.get(include=[])["ids"])
    print(f"  {len(existing_ids):,} existing IDs", flush=True)

    print(f"Scanning {jsonl_path} for new papers ...", flush=True)
    new_papers = []
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            if obj.get("arxiv_id") not in existing_ids:
                new_papers.append(Paper.model_validate(obj))

    print(f"  {total:,} total papers, {len(new_papers):,} new", flush=True)

    if not new_papers:
        print("Nothing to embed. Done.", flush=True)
        return

    print(f"Embedding {len(new_papers):,} new papers -> ChromaDB ...", flush=True)
    dense = DenseRetriever(
        model_name=config["models"]["embedding_model"],
        persist_dir=config["paths"]["dense_index_dir"],
    )
    dense.build_index(new_papers)

    print(f"Done. {len(new_papers):,} papers embedded in {time.time()-t0:.1f}s", flush=True)
    print(f"ChromaDB collection size: {dense.collection.count():,} vectors.", flush=True)


if __name__ == "__main__":
    main()
