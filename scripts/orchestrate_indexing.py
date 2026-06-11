"""
orchestrate_indexing.py

Watches the dense indexer and recovery harvester log files and automatically
runs the remaining pipeline steps once both finish:

    1. merge_recovered_papers.py       -- merge recovered JSONL into main
    2. build_dense_index_fast.py       -- index the recovered papers (idempotent)
    3a. build_bm25_index.py            -- build BM25 index          } run in
    3b. build_keyword_index.py         -- build keyword index        } parallel
    3c. build_metadata_db.py           -- build SQLite metadata index }

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
MERGE_LOG = Path("logs/merge.log")
DENSE2_LOG = Path("logs/build_dense2.log")
BM25_LOG = Path("logs/build_bm25.log")
KEYWORD_LOG = Path("logs/build_keyword.log")
METADATA_DB_LOG = Path("logs/build_metadata_db.log")
ORCH_LOG = Path("logs/orchestrate.log")
RECOVERED_JSONL = Path("data/processed/arxiv_recovered.jsonl")

POLL_INTERVAL = 60  # seconds between log checks


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ORCH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def file_contains(path: Path, marker: str) -> bool:
    try:
        # PowerShell > redirect writes UTF-16 LE (BOM: ff fe); detect and decode correctly.
        with open(path, "rb") as f:
            bom = f.read(2)
        encoding = "utf-16" if bom == b"\xff\xfe" else "utf-8"
        with open(path, "r", encoding=encoding, errors="replace") as f:
            return any(marker in line for line in f)
    except (FileNotFoundError, OSError):
        return False


def run_step(label: str, cmd: list, log_path: Path) -> None:
    log(f"Starting: {label}")
    log(f"Command:  {' '.join(cmd)}")
    log(f"Output:   {log_path}")
    with open(log_path, "w") as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log(f"FAILED (exit {result.returncode}): {label}")
        log("Aborting orchestration. Fix the error and re-run remaining steps manually.")
        sys.exit(result.returncode)
    log(f"Finished: {label}")


def run_parallel(steps: list) -> None:
    """Launch multiple (label, cmd, log_path) steps as subprocesses and wait for all."""
    procs = []
    for label, cmd, log_path in steps:
        log(f"Starting (parallel): {label}")
        log(f"Command:             {' '.join(cmd)}")
        with open(log_path, "w") as out_f:
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT)
        procs.append((label, proc))

    for label, proc in procs:
        proc.wait()
        if proc.returncode != 0:
            log(f"FAILED (exit {proc.returncode}): {label}")
            log("Aborting orchestration. Fix the error and re-run remaining steps manually.")
            sys.exit(proc.returncode)
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
        MERGE_LOG,
    )

    run_step(
        "build_dense_index_fast.py (recovered papers)",
        [sys.executable, "scripts/build_dense_index_fast.py", "--backend", "torch",
         "--jsonl", str(RECOVERED_JSONL)],
        DENSE2_LOG,
    )

    run_parallel([
        (
            "build_bm25_index.py",
            [sys.executable, "scripts/build_bm25_index.py"],
            BM25_LOG,
        ),
        (
            "build_keyword_index.py",
            [sys.executable, "scripts/build_keyword_index.py"],
            KEYWORD_LOG,
        ),
        (
            "build_metadata_db.py",
            [sys.executable, "scripts/build_metadata_db.py"],
            METADATA_DB_LOG,
        ),
    ])

    log("=== All steps complete. Indexing pipeline finished. ===")


if __name__ == "__main__":
    main()
