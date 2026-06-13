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
from src.rag_lit.preprocessing import arxiv_id_sort_key


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

    arxiv_ids: list = []
    corpus_tokens: list = []

    if args.max_papers:
        # Need the full id list up front to select the N most recent papers,
        # so this path still pays the two-list cost (raw text + tokens).
        all_papers = []
        for i, (arxiv_id, text) in enumerate(stream_papers(jsonl_path)):
            all_papers.append((arxiv_id, text))
            if (i + 1) % 200_000 == 0:
                print(f"  Read {i+1:,} papers ({time.time()-t0:.0f}s)", flush=True)

        # Sort by (year, month, sequence) descending so --max-papers selects
        # the most recent papers. A plain lexicographic sort on the id string
        # would put every pre-2007 'category/YYMMNNN' id ahead of every
        # 'YYMM.NNNNN' id (letters > digits in ASCII), which silently drops
        # ~17 years of new-format papers from the "most recent N" selection.
        all_papers.sort(key=lambda x: arxiv_id_sort_key(x[0]), reverse=True)

        if args.max_papers < len(all_papers):
            print(
                f"Keeping {args.max_papers:,} most recent papers "
                f"(dropped {len(all_papers) - args.max_papers:,} older papers)",
                flush=True,
            )
            all_papers = all_papers[: args.max_papers]

        print(f"Tokenizing {len(all_papers):,} papers ...", flush=True)
        for i, (arxiv_id, text) in enumerate(all_papers):
            arxiv_ids.append(arxiv_id)
            corpus_tokens.append([sys.intern(tok) for tok in tokenize(text)])
            if (i + 1) % 100_000 == 0:
                print(f"  Tokenized {i+1:,} papers ({time.time()-t0:.0f}s)", flush=True)
        del all_papers
    else:
        # Full corpus: tokenize while streaming so we never hold both the raw
        # text and the tokenized corpus in memory at the same time. Interning
        # tokens dedupes the (highly repetitive) vocabulary across documents.
        print("Tokenizing while streaming (no --max-papers) ...", flush=True)
        for i, (arxiv_id, text) in enumerate(stream_papers(jsonl_path)):
            arxiv_ids.append(arxiv_id)
            corpus_tokens.append([sys.intern(tok) for tok in tokenize(text)])
            if (i + 1) % 200_000 == 0:
                print(f"  Tokenized {i+1:,} papers ({time.time()-t0:.0f}s)", flush=True)

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
