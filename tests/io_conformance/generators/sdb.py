"""Deterministic SDB (SQLite weather DB) generator for the IO conformance harness.

Uses only the standard-library ``sqlite3`` module so it has no optional backend
dependency.  The schema mirrors the WeeWX/Davis ``archive`` table the SDB reader
expects: an integer ``dateTime`` (Unix seconds) column plus numeric channels.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

__all__ = ["GENERATED_FILES", "generate"]

GENERATED_FILES = ("sample.sdb", "manifest.json")

_START_UNIX = 1_700_000_000
_INTERVAL = 300  # seconds between archive records
_N_RECORDS = 8
_CHANNELS = ("outTemp", "outHumidity", "barometer")


def generate(output_dir: Path) -> dict[str, Path]:
    """Write a tiny deterministic weather SQLite database into *output_dir*."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sdb_path = output_dir / "sample.sdb"
    manifest_path = output_dir / "manifest.json"

    if sdb_path.exists():
        sdb_path.unlink()

    conn = sqlite3.connect(str(sdb_path))
    try:
        conn.execute(
            "CREATE TABLE archive ("
            "  dateTime INTEGER,"
            "  outTemp REAL,"
            "  outHumidity REAL,"
            "  barometer REAL"
            ")"
        )
        for i in range(_N_RECORDS):
            conn.execute(
                "INSERT INTO archive VALUES (?, ?, ?, ?)",
                (
                    _START_UNIX + i * _INTERVAL,
                    70.0 + i,
                    50.0 + i * 0.5,
                    29.92,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    manifest_path.write_text(
        json.dumps(
            {
                "generator": "sdb",
                "files": list(GENERATED_FILES),
                "table": "archive",
                "channels": list(_CHANNELS),
                "start_unix": _START_UNIX,
                "interval_s": _INTERVAL,
                "n_records": _N_RECORDS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "sdb": sdb_path,
        "manifest": manifest_path,
    }
