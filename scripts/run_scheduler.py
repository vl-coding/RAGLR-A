"""
run_scheduler.py

Runs incremental_update.py twice a day at 00:00 and 12:00 UTC.
Start this as a persistent background process alongside the pipeline server.

Usage:
    python scripts/run_scheduler.py
    python scripts/run_scheduler.py --hours 0 6 12 18   # four times a day
    python scripts/run_scheduler.py --log logs/scheduler.log
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_HOURS_UTC = [0, 12]


def next_run_time(hours_utc: list) -> datetime:
    now = datetime.now(timezone.utc)
    base = now.replace(minute=0, second=0, microsecond=0)
    for hour in sorted(hours_utc):
        candidate = base.replace(hour=hour)
        if candidate > now:
            return candidate
    # All today's slots passed — first slot tomorrow
    tomorrow = (now + timedelta(days=1)).replace(
        hour=sorted(hours_utc)[0], minute=0, second=0, microsecond=0
    )
    return tomorrow


def log(msg: str, log_path: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours", nargs="+", type=int, default=DEFAULT_HOURS_UTC,
        help="UTC hours to run (default: 0 12)",
    )
    parser.add_argument(
        "--log", default="logs/scheduler.log",
        help="Path to scheduler log file (default: logs/scheduler.log)",
    )
    args = parser.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    update_script = str(Path(__file__).parent / "incremental_update.py")

    log(f"Scheduler started. Runs at hours {sorted(args.hours)} UTC.", args.log)

    while True:
        run_at = next_run_time(args.hours)
        wait = (run_at - datetime.now(timezone.utc)).total_seconds()
        log(f"Next update at {run_at.isoformat(timespec='seconds')} UTC ({wait/3600:.1f}h away)", args.log)

        time.sleep(max(0.0, wait))

        log("Starting incremental update ...", args.log)
        result = subprocess.run([sys.executable, update_script])

        if result.returncode == 0:
            log("Update completed successfully.", args.log)
        else:
            log(f"Update FAILED (exit {result.returncode}) — will retry at next scheduled time.", args.log)


if __name__ == "__main__":
    main()
