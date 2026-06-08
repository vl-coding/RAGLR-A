"""
orchestrate_indexing.py

Watches the dense indexer and recovery harvester log files and automatically
runs the remaining pipeline steps once both finish:

    1. merge_recovered_papers.py       -- merge recovered JSONL into main
    2. build_dense_index_fast.py       -- index the recovered papers (idempotent)
    3. build_indexes.py --skip-dense   -- build BM25 + keyword indexes

Safe to run alongside the already-running jobs -- it only reads their log
files and does not touch any data files until both jobs are confirmed done.

Usage:
    python scripts/orchestrate_indexing.py
"""

import subprocess
import sys
import time
from pathlib import Path

DENSE_LOG = Path("logs/build_dense.log")
RECOVER_LOG = Path("logs/recover_papers.log")
ORCH_LOG = Path("logs/orchestrate.log")

POLL_INTERVAL = 60  # seconds between log checks


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ORCH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def file_contains(path: Path, marker: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return any(marker in line for line in f)
    except FileNotFoundError:
        return False


def run_step(label: str, cmd: list) -> None:
    log(f"Starting: {label}")
    log(f"Command:  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log(f"FAILED (exit {result.returncode}): {label}")
        log("Aborting orchestration. Fix the error and re-run remaining steps manually.")
        sys.exit(result.returncode)
    log(f"Finished: {label}")


def wait_for(log_path: Path, marker: str, label: str) -> None:
    log(f"Waiting for {label} (watching {log_path} for '{marker}') ...")
    while not file_contains(log_path, marker):
        time.sleep(POLL_INTERVAL)
    log(f"Done detected: {label}")


def main() -> None:
    log("=== Orchestrator started ===")
    log(f"Watching dense indexer:    {DENSE_LOG}")
    log(f"Watching recovery harvest: {RECOVER_LOG}")
    log(f"Poll interval: {POLL_INTERVAL}s")

    # Wait for dense indexer (fires first, ~5h)
    wait_for(DENSE_LOG, "Done.", "dense indexer")

    # Wait for recovery harvester (fires second, ~7h)
    wait_for(RECOVER_LOG, "Done. Recovered", "recovery harvester")

    log("Both jobs complete. Starting pipeline sequence.")

    run_step(
        "merge_recovered_papers.py",
        [sys.executable, "scripts/merge_recovered_papers.py"],
    )

    run_step(
        "build_dense_index_fast.py (recovered papers)",
        [sys.executable, "scripts/build_dense_index_fast.py", "--backend", "torch"],
    )

    run_step(
        "build_indexes.py --skip-dense (BM25 + keyword)",
        [sys.executable, "scripts/build_indexes.py", "--skip-dense"],
    )

    log("=== All steps complete. Indexing pipeline finished. ===")


if __name__ == "__main__":
    main()
