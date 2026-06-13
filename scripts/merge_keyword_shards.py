import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lit.config import load_config, ensure_project_dirs

SHARD_COUNT = 13


def main():
    t0 = time.time()
    config = load_config()
    ensure_project_dirs(config)

    out_path = Path(config["paths"]["keyword_index"])
    shard_paths = [
        out_path.with_name(f"{out_path.name}.shard{i}.tmp") for i in range(SHARD_COUNT)
    ]
    for p in shard_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    tmp_path = str(out_path) + ".tmp"
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("CREATE TABLE postings (token TEXT PRIMARY KEY, arxiv_ids TEXT NOT NULL)")
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
