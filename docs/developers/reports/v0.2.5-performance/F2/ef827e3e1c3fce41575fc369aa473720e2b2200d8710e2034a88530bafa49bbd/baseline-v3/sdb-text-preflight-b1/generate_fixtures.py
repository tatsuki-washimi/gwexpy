"""Generate fixed public-parity fixtures for the SDB TEXT preflight review."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FIXTURE_DIR = ROOT / "fixtures"
TIMESTAMPS = [1_700_000_000, 1_700_000_300, 1_700_000_600]

# One selected sample (row 1) and two rows outside the requested window make
# selected-value behavior distinguishable from the legacy whole-source scan.
CASES: dict[str, dict[str, Any]] = {
    "integer_text_clean": {"column": "outTemp", "values": ["60", "61", "62"]},
    "decimal_text": {"column": "outTemp", "values": ["60", "61.5", "62"]},
    "whitespace_integer_text": {
        "column": "outTemp",
        "values": ["60", " 61 ", "62"],
    },
    "mixed_integer_real_null": {
        "column": "outTemp",
        "values": [60, 61.5, None],
    },
    "invalid_selected_text": {
        "column": "outTemp",
        "values": ["60", "not-a-number", "62"],
    },
    "invalid_outside_text": {
        "column": "outTemp",
        "values": ["not-a-number", "61", "62"],
    },
    "invalid_outside_irregular_timestamp": {
        "column": "outTemp",
        "values": ["not-a-number", "61", "62"],
        "timestamps": [1_700_000_000, 1_700_000_300, 1_700_000_601],
    },
    "int64_uint64_edges_text": {
        "column": "outTemp",
        "values": [
            "-9223372036854775808",
            "18446744073709551615",
            "9223372036854775807",
        ],
    },
    "float64_integer_limit_text": {
        "column": "outHumidity",
        "values": [
            "9007199254740990",
            "9007199254740991",
            "9007199254740992",
        ],
    },
    "float64_integer_above_limit_text": {
        "column": "outHumidity",
        "values": ["60", "9007199254740993", "62"],
    },
    "overflow_outside_text": {
        "column": "barometer",
        "values": ["60", "61", "1e308"],
    },
    "overflow_selected_text": {
        "column": "barometer",
        "values": ["60", "1e308", "62"],
    },
    "underflow_outside_text": {
        "column": "windSpeed",
        "values": ["60", "61", "1e-320"],
    },
    "underflow_selected_text": {
        "column": "windSpeed",
        "values": ["60", "1e-320", "62"],
    },
    "blob_outside": {"column": "outTemp", "values": [b"\xff", "61", "62"]},
    "blob_selected": {"column": "outTemp", "values": ["60", b"\xff", "62"]},
    "null_outside": {"column": "outTemp", "values": [None, "61", "62"]},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for case_name, case in CASES.items():
        path = FIXTURE_DIR / f"{case_name}.sdb"
        path.unlink(missing_ok=True)
        column = case["column"]
        with sqlite3.connect(path) as connection:
            connection.execute(
                f'CREATE TABLE archive (dateTime INTEGER, "{column}")'
            )
            connection.executemany(
                "INSERT INTO archive VALUES (?, ?)",
                zip(case.get("timestamps", TIMESTAMPS), case["values"]),
            )
            storage = connection.execute(
                f'SELECT typeof("{column}") FROM archive ORDER BY rowid'
            ).fetchall()
        files[case_name] = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "column": column,
            "source_values": [
                value.decode("latin1") if isinstance(value, bytes) else value
                for value in case["values"]
            ],
            "storage_classes": [str(row[0]) for row in storage],
            "timestamps_unix": case.get("timestamps", TIMESTAMPS),
            "selected_row_index": 1,
        }
    manifest = {
        "schema": 1,
        "generator": "generate_fixtures.py",
        "rows_per_fixture": len(TIMESTAMPS),
        "window_rows": [1, 2],
        "cases": files,
    }
    (ROOT / "fixture-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
