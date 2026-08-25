"""Run explicit pytest nodes one at a time and persist JSONL comparison evidence."""

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
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--doctest-modules", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    nodes = [
        node for node in args.nodes.read_text(encoding="utf-8").splitlines() if node
    ]
    logs = args.logs.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ | {
        "PYTHONPATH": str(root),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "MPLCONFIGDIR": "/tmp/gwexpy-v020-proxy-remediation-doctest-node-mpl",
    }
    with args.output.resolve().open("w", encoding="utf-8") as stream:
        for index, node in enumerate(nodes, start=1):
            log = logs / f"{index:03d}.log"
            command = [sys.executable, "-m", "pytest", "-q"]
            if args.doctest_modules:
                command.append("--doctest-modules")
            command.append(node)
            started = time.monotonic()
            try:
                result = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                log.write_text(result.stdout + result.stderr, encoding="utf-8")
                status = "pass" if result.returncode == 0 else "fail"
                exit_status: int | None = result.returncode
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout or ""
                stderr = error.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                log.write_text(stdout + stderr, encoding="utf-8")
                status = "timeout"
                exit_status = None
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "node": node,
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
