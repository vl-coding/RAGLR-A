import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.keyword_index import save_keyword_index_db, tokenize


def main():
    t0 = time.time()
    config = load_config()
    ensure_project_dirs(config)

    jsonl_path = config["data"]["processed_path"]
    print(f"Streaming papers from {jsonl_path} ...", flush=True)

    index = defaultdict(set)
    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            arxiv_id = obj.get("arxiv_id", "")
            title = obj.get("title", "")
            abstract = obj.get("abstract", "")
            text = f"{title}\n\n{abstract}"
            for token in set(tokenize(text)):
                index[token].add(arxiv_id)
            count += 1
            if count % 100_000 == 0:
                print(f"  Indexed {count:,} papers, {len(index):,} tokens ({time.time()-t0:.0f}s)", flush=True)

    print(f"Indexed {count:,} papers -> {len(index):,} unique tokens in {time.time()-t0:.1f}s", flush=True)

    out = config["paths"]["keyword_index"]
    save_keyword_index_db(dict(index), out)
    print(f"Saved to {out}", flush=True)
    print(f"Done. Total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
