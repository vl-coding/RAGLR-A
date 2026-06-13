import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.keyword_index import tokenize

# Papers per shard. At ~1.2M papers the full in-memory defaultdict(set) hit
# ~5GB, which doesn't fit alongside the rest of this 16GB machine's baseline
# usage for the full 3.07M-paper corpus. Sharding keeps each in-memory dict
# to a fraction of that, writing each shard's postings to its own temp SQLite
# table and merging them on disk afterwards.
SHARD_SIZE = 250_000


def write_shard(index: dict, shard_path: Path) -> None:
    conn = sqlite3.connect(str(shard_path))
    try:
        conn.execute("CREATE TABLE postings (token TEXT PRIMARY KEY, arxiv_ids TEXT NOT NULL)")
        batch = []
        for token, ids in index.items():
            batch.append((token, ",".join(ids)))
            if len(batch) >= 50_000:
                conn.executemany("INSERT INTO postings (token, arxiv_ids) VALUES (?, ?)", batch)
                batch.clear()
        if batch:
            conn.executemany("INSERT INTO postings (token, arxiv_ids) VALUES (?, ?)", batch)
        conn.commit()
    finally:
        conn.close()


def main():
    t0 = time.time()
    config = load_config()
    ensure_project_dirs(config)

    jsonl_path = config["data"]["processed_path"]
    out_path = Path(config["paths"]["keyword_index"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming papers from {jsonl_path} (sharded, {SHARD_SIZE:,} papers/shard) ...", flush=True)

    shard_paths: list[Path] = []
    index: dict = defaultdict(set)
    count = 0

    def flush_shard():
        nonlocal index
        shard_path = out_path.with_name(f"{out_path.name}.shard{len(shard_paths)}.tmp")
        if shard_path.exists():
            shard_path.unlink()
        write_shard(index, shard_path)
        shard_paths.append(shard_path)
        print(
            f"  Wrote shard {len(shard_paths)-1} ({len(index):,} tokens) at {time.time()-t0:.0f}s",
            flush=True,
        )
        index = defaultdict(set)

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
            if count % SHARD_SIZE == 0:
                flush_shard()
            elif count % 100_000 == 0:
                print(f"  Read {count:,} papers ({time.time()-t0:.0f}s)", flush=True)

    if index:
        flush_shard()

    print(f"Read {count:,} papers in {len(shard_paths)} shards, {time.time()-t0:.1f}s. Merging ...", flush=True)

    tmp_path = str(out_path) + ".tmp"
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("CREATE TABLE postings (token TEXT PRIMARY KEY, arxiv_ids TEXT NOT NULL)")
        # Merge shards one at a time (rather than ATTACHing all of them, which
        # hits SQLite's default 10-attached-database limit for >10 shards).
        for i, shard_path in enumerate(shard_paths):
            conn.execute("ATTACH DATABASE ? AS shard", (str(shard_path),))
            # Append to existing tokens, then insert tokens new to this shard.
            conn.execute(
                "UPDATE postings SET arxiv_ids = arxiv_ids || ',' || "
                "(SELECT s.arxiv_ids FROM shard.postings s WHERE s.token = postings.token) "
                "WHERE token IN (SELECT token FROM shard.postings)"
            )
            conn.execute(
                "INSERT INTO postings (token, arxiv_ids) "
                "SELECT token, arxiv_ids FROM shard.postings "
                "WHERE token NOT IN (SELECT token FROM postings)"
            )
            conn.commit()
            conn.execute("DETACH DATABASE shard")
            print(f"  Merged shard {i}/{len(shard_paths)-1} at {time.time()-t0:.0f}s", flush=True)
    finally:
        conn.close()

    for shard_path in shard_paths:
        shard_path.unlink()

    if out_path.exists():
        out_path.unlink()
    Path(tmp_path).rename(out_path)

    print(f"Saved to {out_path}", flush=True)
    print(f"Done. Total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
