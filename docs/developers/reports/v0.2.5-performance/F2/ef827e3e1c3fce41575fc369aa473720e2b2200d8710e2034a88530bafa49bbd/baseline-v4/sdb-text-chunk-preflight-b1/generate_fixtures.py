"""Generate v4 B1 parity fixtures crossing bounded SDB scan chunks."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FIXTURE_DIR = ROOT / "fixtures"
ROW_COUNT = 600
FIRST_UNIX = 1_700_000_000
SAMPLE_DT = 300

CASES: dict[str, dict[str, Any]] = {
    "mixed_text_real_null_near_chunk_boundary": {
        "overrides": {3: 62.5, 17: None, 255: 100.25, 256: None, 511: 70.75, 512: None},
        "window_rows": [254, 260],
    },
    "invalid_text_at_chunk_boundary_outside_window": {
        "overrides": {
            3: 62.5,
            17: None,
            255: 100.25,
            256: "not-a-number",
            511: 70.75,
            512: None,
        },
        "window_rows": [500, 504],
    },
    "trailing_newline_integer_text_at_chunk_boundary": {
        "overrides": {3: 62.5, 17: None, 255: "61\n", 256: None},
        "window_rows": [253, 258],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(overrides: dict[int, Any]) -> list[Any]:
    values: list[Any] = [str(60 + index % 20) for index in range(ROW_COUNT)]
    for index, value in overrides.items():
        values[index] = value
    return values


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    case_manifest: dict[str, Any] = {}
    for name, case in CASES.items():
        path = FIXTURE_DIR / f"{name}.sdb"
        path.unlink(missing_ok=True)
        values = _values(case["overrides"])
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp)")
            connection.executemany(
                "INSERT INTO archive VALUES (?, ?)",
                [
                    (FIRST_UNIX + index * SAMPLE_DT, value)
                    for index, value in enumerate(values)
                ],
            )
            storage_classes = [
                str(row[0])
                for row in connection.execute(
                    'SELECT typeof("outTemp") FROM archive ORDER BY rowid'
                )
            ]
        window_start_row, window_end_row = case["window_rows"]
        case_manifest[name] = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "column": "outTemp",
            "rows": ROW_COUNT,
            "source_sample_dt": SAMPLE_DT,
            "storage_class_counts": dict(sorted(Counter(storage_classes).items())),
            "overrides": {
                str(index): value for index, value in case["overrides"].items()
            },
            "boundary_values_254_258": values[254:259],
            "selected_window_rows_half_open": [window_start_row, window_end_row],
            "window_unix": {
                "start": FIRST_UNIX + window_start_row * SAMPLE_DT,
                "end": FIRST_UNIX + window_end_row * SAMPLE_DT,
            },
        }
    manifest = {
        "schema": 1,
        "generator": "generate_fixtures.py",
        "rows_per_fixture": ROW_COUNT,
        "chunk_rows_under_review": 256,
        "cases": case_manifest,
    }
    (ROOT / "fixture-manifest-v4.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
