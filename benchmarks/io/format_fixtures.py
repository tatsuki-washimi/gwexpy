"""Package-independent SDB and TDMS fixtures for the v0.2.5 F2 benchmark.

Only fixture writers are imported here. In particular, this module never imports
gwexpy, so the same files can be read by the B0, B1, and candidate wheels.
The returned manifests contain relative file names and SHA-256 hashes; callers
should record these manifests before measuring any reader.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_START_UNIX = 1_700_000_000
_INTERVAL_SECONDS = 300
_SDB_COLUMNS = ("outTemp", "outHumidity", "barometer")
_TDMS_CHANNELS = ("Selected", "Other", "Third")


def _file_facts(path: Path, directory: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.relative_to(directory).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def _seal_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Attach a stable digest of all recorded fixture facts."""
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def _write_sdb(
    path: Path,
    rows: int,
    *,
    bad_units: bool = False,
    bad_time: bool = False,
    bad_column: str | None = None,
    wal: bool = False,
) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA page_size=4096")
        connection.execute(
            "CREATE TABLE archive (dateTime INTEGER, outTemp TEXT, "
            "outHumidity TEXT, barometer TEXT, usUnits)"
        )
        batch = []
        for index in range(rows):
            values: dict[str, int | str] = {
                "outTemp": str(60 + index % 25),
                "outHumidity": str(40 + index % 40),
                "barometer": str(29 + (index % 10) / 100),
            }
            if index == rows - 1 and bad_column is not None:
                values[bad_column] = "not-a-number"
            date_time = _START_UNIX + index * _INTERVAL_SECONDS
            if index == rows - 1 and bad_time:
                date_time += 1
            batch.append(
                (
                    date_time,
                    values["outTemp"],
                    values["outHumidity"],
                    values["barometer"],
                    2 if index == rows - 1 and bad_units else 1,
                )
            )
            if len(batch) == 1024:
                connection.executemany(
                    "INSERT INTO archive VALUES (?, ?, ?, ?, ?)", batch
                )
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO archive VALUES (?, ?, ?, ?, ?)", batch)
        connection.commit()
        if wal:
            connection.execute("PRAGMA journal_mode=WAL")


def make_sdb_fixtures(directory: str | Path, *, rows: int = 4096) -> dict[str, Any]:
    """Write deterministic SDB cases and return a JSON-ready manifest.

    The window covers the first eighth of the file. Both table-wide faults are
    deliberately in the last row, outside that window. ``read_kwargs`` are
    arguments for the SDB dict reader; ``window_unix`` gives the matching
    half-open interval in Unix seconds for callers that test time selection.
    """
    if rows < 16:
        raise ValueError("SDB fixtures require at least 16 rows")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    selected_rows = rows // 8
    cases: dict[str, dict[str, Any]] = {}
    definitions = {
        "valid_large": {},
        "fault_usunits_unselected_window": {"bad_units": True},
        "fault_irregular_time_unselected_window": {"bad_time": True},
        "fault_bad_payload_unselected": {"bad_column": "outHumidity"},
        "fault_bad_payload_selected": {"bad_column": "outTemp"},
        "wal_base": {"wal": True},
    }
    for case_name, options in definitions.items():
        path = root / f"sdb_{case_name}.sdb"
        _write_sdb(path, rows, **options)
        cases[case_name] = {
            **_file_facts(path, root),
            "rows": rows,
            "columns": list(_SDB_COLUMNS),
            "selected_columns": ["outTemp"],
            "selected_rows": rows,
            "window_selected_rows": selected_rows,
            "read_kwargs": {"columns": ["outTemp"]},
            "window_unix": {
                "start": _START_UNIX,
                "end": _START_UNIX + selected_rows * _INTERVAL_SECONDS,
            },
            "fault": (
                "usUnits_last_row"
                if options.get("bad_units")
                else "irregular_dateTime_last_row"
                if options.get("bad_time")
                else f"non_numeric_{options['bad_column']}_last_row"
                if options.get("bad_column")
                else None
            ),
        }
    return _seal_manifest(
        {
            "format": "sdb",
            "start_unix": _START_UNIX,
            "interval_seconds": _INTERVAL_SECONDS,
            "cases": cases,
        }
    )


class SdbWalSnapshot:
    """Two-connection WAL schedule with explicit synchronization points.

    ``begin_reader_snapshot`` pins the supplied SQLite reader connection.
    A test can route its package reader through that connection if needed.
    No scheduler timing or sleep is involved.
    """

    def __init__(self, reader: sqlite3.Connection, writer: sqlite3.Connection):
        self.reader = reader
        self.writer = writer
        self._begun = False
        self._committed = False

    def begin_reader_snapshot(self) -> str:
        """Pin the first-row snapshot and return its original temperature."""
        if self._begun:
            raise RuntimeError("reader snapshot has already begun")
        self.reader.execute("BEGIN")
        value = self.reader.execute(
            "SELECT outTemp FROM archive ORDER BY rowid LIMIT 1"
        ).fetchone()
        self._begun = True
        assert value is not None
        return str(value[0])

    def commit_writer_update(self, value: str = "99") -> None:
        """Commit a first-row payload change while the reader stays pinned."""
        if not self._begun or self._committed:
            raise RuntimeError("begin one reader snapshot before one writer update")
        self.writer.execute("BEGIN IMMEDIATE")
        self.writer.execute("UPDATE archive SET outTemp = ? WHERE rowid = 1", (value,))
        self.writer.commit()
        self._committed = True

    def observed_values(self) -> tuple[str, str]:
        """Return values seen by the pinned reader and the committed writer."""
        if not self._committed:
            raise RuntimeError("writer update has not committed")
        query = "SELECT outTemp FROM archive ORDER BY rowid LIMIT 1"
        old = self.reader.execute(query).fetchone()
        new = self.writer.execute(query).fetchone()
        assert old is not None and new is not None
        return str(old[0]), str(new[0])


@contextmanager
def open_sdb_wal_snapshot(path: str | Path) -> Iterator[SdbWalSnapshot]:
    """Open independent WAL connections for a reproducible concurrent update.

    The caller controls when the reader snapshot starts and when the writer
    commits. The context closes both connections; an update deliberately
    persists, so copy ``wal_base`` before repeating a benchmark run.
    """
    source = Path(path)
    reader = sqlite3.connect(source, isolation_level=None)
    writer = sqlite3.connect(source, isolation_level=None)
    try:
        mode = writer.execute("PRAGMA journal_mode=WAL").fetchone()
        if mode is None or str(mode[0]).lower() != "wal":
            raise RuntimeError("SQLite WAL mode is unavailable")
        schedule = SdbWalSnapshot(reader, writer)
        yield schedule
    finally:
        if reader.in_transaction:
            reader.rollback()
        reader.close()
        writer.close()


def _write_tdms(
    path: Path,
    rows: int,
    channel_order: tuple[str, ...],
    bad_increment: str | None = None,
) -> None:
    import numpy as np
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    if path.exists():
        path.unlink()
    objects = [RootObject(), GroupObject("Group")]
    for index, name in enumerate(channel_order):
        properties = {"wf_start_time": 0.0}
        if name != bad_increment:
            properties["wf_increment"] = 0.25
        data = np.arange(rows, dtype=np.float64) + index * 1000.0
        objects.append(ChannelObject("Group", name, data, properties=properties))
    with TdmsWriter(str(path)) as writer:
        writer.write_segment(objects)


def make_tdms_truncated_payload(path: str | Path, *, bytes_to_remove: int = 8) -> None:
    """Shorten a TDMS file's trailing raw-data segment by one float64 sample.

    A reader may reject this during file open or during channel data access;
    that stage depends on npTDMS internals. The last channel in the source
    file is the intended damaged channel. This operation makes no assumption
    about the TDMS metadata encoding and works on files written by npTDMS.
    """
    target = Path(path)
    size = target.stat().st_size
    if bytes_to_remove <= 0 or bytes_to_remove >= size:
        raise ValueError("bytes_to_remove must be positive and smaller than the file")
    with target.open("r+b") as stream:
        stream.truncate(size - bytes_to_remove)


def make_tdms_fixtures(directory: str | Path, *, rows: int = 4096) -> dict[str, Any]:
    """Write multi-channel TDMS selection and fault cases with file hashes.

    Requires optional ``nptdms`` and NumPy only when called. Truncated files
    exercise malformed raw payload, but a cross-version guarantee that the
    fault is isolated to one channel is impossible without a TDMS parser.
    The manifest records that limitation for fault-matrix interpretation.
    """
    if rows < 2:
        raise ValueError("TDMS fixtures require at least two samples per channel")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict[str, Any]] = {}
    definitions = {
        "valid_many": (_TDMS_CHANNELS, None, False),
        "fault_selected_increment": (_TDMS_CHANNELS, "Selected", False),
        "fault_unselected_increment": (_TDMS_CHANNELS, "Other", False),
        "fault_unselected_payload": (("Selected", "Third", "Other"), None, True),
        "fault_selected_payload": (("Other", "Third", "Selected"), None, True),
    }
    for case_name, (order, bad_increment, truncate) in definitions.items():
        path = root / f"tdms_{case_name}.tdms"
        _write_tdms(path, rows, order, bad_increment)
        if truncate:
            make_tdms_truncated_payload(path)
        cases[case_name] = {
            **_file_facts(path, root),
            "rows": rows,
            "channels": [f"Group/{name}" for name in order],
            "selected_channels": ["Group/Selected"],
            "selected_rows": None if case_name == "fault_selected_payload" else rows,
            "nominal_selected_rows": rows,
            "read_kwargs": {"channels": ["Group/Selected"]},
            "fault": (
                f"missing_wf_increment_{bad_increment}"
                if bad_increment is not None
                else f"truncated_raw_payload_{order[-1]}"
                if truncate
                else None
            ),
            "payload_fault_scope": (
                "trailing-channel-intended; reader may fail at file open"
                if truncate
                else None
            ),
        }
    return _seal_manifest(
        {"format": "tdms", "sample_interval_seconds": 0.25, "cases": cases}
    )
