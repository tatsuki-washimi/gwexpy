"""Run deterministic per-file timeseries pytest audits and write JSONL evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    logs = args.logs.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    test_files = sorted((root / "tests/timeseries").glob("test_*.py"))
    environment = os.environ | {"PYTHONPATH": str(root)}

    with output.open("w", encoding="utf-8") as stream:
        for index, test_file in enumerate(test_files, start=1):
            relative = test_file.relative_to(root)
            log = logs / f"{index:03d}-{test_file.stem}.log"
            started = time.monotonic()
            status = "pass"
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", str(relative)],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                log.write_text(result.stdout + result.stderr, encoding="utf-8")
                if result.returncode:
                    status = "fail"
                exit_status: int | None = result.returncode
            except subprocess.TimeoutExpired as error:
                status = "timeout"
                exit_status = None
                stdout = error.stdout or ""
                stderr = error.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                log.write_text(stdout + stderr, encoding="utf-8")
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "test_file": str(relative),
                "status": status,
                "exit_status": exit_status,
                "duration_seconds": round(time.monotonic() - started, 3),
                "timeout_seconds": args.timeout,
                "log": str(log),
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
