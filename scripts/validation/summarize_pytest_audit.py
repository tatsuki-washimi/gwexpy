"""Summarize JSONL pytest audit records, including per-file skip counts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SUMMARY_COUNTS = re.compile(
    r"(?P<count>\d+) (?P<kind>passed|failed|skipped|xfailed|xpassed)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def counts_from_log(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for count, kind in SUMMARY_COUNTS.findall(path.read_text(encoding="utf-8")):
        counts[kind] += int(count)
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    results = []
    totals: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for record in records:
        result = record | {"pytest_counts": counts_from_log(Path(record["log"]))}
        results.append(result)
        statuses[result["status"]] += 1
        totals.update(result["pytest_counts"])
    args.output.write_text(
        json.dumps(
            {
                "audit_file": str(args.input),
                "file_count": len(results),
                "status_counts": dict(sorted(statuses.items())),
                "pytest_totals": dict(sorted(totals.items())),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
