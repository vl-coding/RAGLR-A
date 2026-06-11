import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import bm25s

from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.keyword_index import tokenize


def stream_papers(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            arxiv_id = obj.get("arxiv_id", "")
            title = obj.get("title", "")
            abstract = obj.get("abstract", "")
            yield arxiv_id, f"{title}\n\n{abstract}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Index only the N most recent papers (by arxiv_id). "
             "Use ~1_000_000 if RAM is limited (3M papers requires ~6GB free RAM).",
    )
    args = parser.parse_args()

    t0 = time.time()
    config = load_config()
    ensure_project_dirs(config)

    jsonl_path = config["data"]["processed_path"]
    print(f"Streaming papers from {jsonl_path} ...", flush=True)

    all_papers = []
    for i, (arxiv_id, text) in enumerate(stream_papers(jsonl_path)):
        all_papers.append((arxiv_id, text))
        if (i + 1) % 200_000 == 0:
            print(f"  Read {i+1:,} papers ({time.time()-t0:.0f}s)", flush=True)

    # Sort by arxiv_id descending so --max-papers selects the most recent.
    # arxiv IDs are zero-padded and encode submission date, so lexicographic
    # sort is chronological.
    all_papers.sort(key=lambda x: x[0], reverse=True)

    if args.max_papers and args.max_papers < len(all_papers):
        print(
            f"Keeping {args.max_papers:,} most recent papers "
            f"(dropped {len(all_papers) - args.max_papers:,} older papers)",
            flush=True,
        )
        all_papers = all_papers[: args.max_papers]

    print(f"Tokenizing {len(all_papers):,} papers ...", flush=True)
    arxiv_ids = []
    corpus_tokens = []
    for i, (arxiv_id, text) in enumerate(all_papers):
        arxiv_ids.append(arxiv_id)
        corpus_tokens.append(tokenize(text))
        if (i + 1) % 100_000 == 0:
            print(f"  Tokenized {i+1:,} papers ({time.time()-t0:.0f}s)", flush=True)

    del all_papers
    print(f"Loaded {len(arxiv_ids):,} papers in {time.time()-t0:.1f}s", flush=True)

    print("Building BM25 index ...", flush=True)
    t1 = time.time()
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    del corpus_tokens
    print(f"BM25 built in {time.time()-t1:.1f}s", flush=True)

    out = config["paths"]["bm25_index"]
    Path(out).mkdir(parents=True, exist_ok=True)
    bm25.save(str(Path(out) / "index"))
    np.save(str(Path(out) / "arxiv_ids.npy"), np.array(arxiv_ids, dtype=object))
    print(f"Saved to {out}", flush=True)
    print(f"Done. Total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
