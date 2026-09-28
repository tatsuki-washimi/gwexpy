"""Reproduce SDB WAL mutations at the old-R payload SELECT boundary.

This is a correctness fault matrix. The F2 baseline-v1 harness remains the
source for SDB timing, memory, and fetched-row structural comparisons.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any

try:
    from . import run
    from .format_fixtures import make_sdb_fixtures
except ImportError:  # Direct execution under python -I.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run
    from format_fixtures import make_sdb_fixtures


MUTATIONS: dict[str, tuple[str, tuple[Any, ...]]] = {
    "selected_value": ("UPDATE archive SET outTemp = ? WHERE rowid = 1", ("99",)),
    "unselected_value": (
        "UPDATE archive SET outHumidity = ? WHERE rowid = 1",
        ("99",),
    ),
    "selected_malformed": (
        "UPDATE archive SET outTemp = ? WHERE rowid = 1",
        ("not-a-number",),
    ),
    "unselected_malformed": (
        "UPDATE archive SET outHumidity = ? WHERE rowid = 1",
        ("not-a-number",),
    ),
    "timestamp": ("UPDATE archive SET dateTime = dateTime + 1 WHERE rowid = 1", ()),
    "usunits": ("UPDATE archive SET usUnits = 2 WHERE rowid = 1", ()),
    "schema": ("ALTER TABLE archive RENAME COLUMN outTemp TO outTempRenamed", ()),
}
SELECTIONS = ("selected", "all")
ORDER = ("A", "B", "B", "A", "B", "A", "A", "B", "A", "B")
ROW_SQL = "SELECT dateTime, outTemp, outHumidity, usUnits FROM archive WHERE rowid = 1"


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _schema(connection: sqlite3.Connection) -> list[str]:
    return [str(row[1]) for row in connection.execute("PRAGMA table_info(archive)")]


def _row(connection: sqlite3.Connection, columns: list[str]) -> list[Any]:
    if "outTemp" not in columns:
        return list(
            connection.execute(
                "SELECT dateTime, outTempRenamed, outHumidity, usUnits "
                "FROM archive WHERE rowid = 1"
            ).fetchone()
        )
    return list(connection.execute(ROW_SQL).fetchone())


def _read_scheduled(source: Path, mutation: str, selection: str) -> dict[str, Any]:
    from gwexpy.timeseries import TimeSeriesDict

    events: dict[str, Any] = {
        "triggered": False,
        "reader_sql_before_trigger": [],
        "mutation": mutation,
        "selection": selection,
    }
    sql_update, params = MUTATIONS[mutation]
    with tempfile.TemporaryDirectory(prefix="gwexpy-sdb-wal-v2-") as scratch:
        path = Path(scratch) / "schedule.sdb"
        shutil.copyfile(source, path)
        original_connect = sqlite3.connect

        class TracedCursor(sqlite3.Cursor):
            def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                sql_text = str(sql).casefold()
                if not events["triggered"]:
                    if (
                        "select" in sql_text
                        and "outtemp" in sql_text
                        and "datetime" in sql_text
                        and "from" in sql_text
                    ):
                        events["trigger_sql"] = str(sql)
                        writer = original_connect(path)
                        try:
                            wal_mode = writer.execute(
                                "PRAGMA journal_mode=WAL"
                            ).fetchone()
                            events["wal_mode"] = wal_mode[0] if wal_mode else None
                            events["schema_before"] = _schema(writer)
                            events["row_before"] = _row(writer, events["schema_before"])
                            writer.execute("BEGIN IMMEDIATE")
                            writer.execute(sql_update, params)
                            writer.commit()
                            events["writer_sql"] = [
                                "BEGIN IMMEDIATE",
                                sql_update,
                                "COMMIT",
                            ]
                            events["writer_parameters"] = list(params)
                            events["writer_committed"] = True
                            events["schema_after"] = _schema(writer)
                            events["row_after"] = _row(writer, events["schema_after"])
                        finally:
                            writer.close()
                        events["triggered"] = True
                    else:
                        events["reader_sql_before_trigger"].append(str(sql))
                return super().execute(sql, *args, **kwargs)

        class TracedConnection(sqlite3.Connection):
            def cursor(self, *args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("factory", TracedCursor)
                return super().cursor(*args, **kwargs)

        def traced_connect(database: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("factory", TracedConnection)
            return original_connect(database, *args, **kwargs)

        sqlite3.connect = traced_connect
        try:
            with (
                warnings.catch_warnings(record=True) as caught,
                run._capture_route_logs(True) as logs,
            ):
                warnings.simplefilter("always")
                try:
                    options = (
                        {"columns": ["outTemp"]} if selection == "selected" else {}
                    )
                    result = TimeSeriesDict.read(path, format="sdb", **options)
                    outcome: dict[str, Any] = {
                        "outcome": "return",
                        "fingerprint": run._fingerprint(result),
                    }
                except Exception as exc:
                    outcome = {
                        "outcome": "error",
                        "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
            outcome["warnings"] = [
                {
                    "category": f"{item.category.__module__}.{item.category.__qualname__}",
                    "message": str(item.message),
                }
                for item in caught
            ]
            outcome["logs"] = logs
            outcome["events"] = events
            return outcome
        finally:
            sqlite3.connect = original_connect


def _worker(args: argparse.Namespace) -> None:
    audit = run._worker_audit(Path(args.wheel), args.version, full=args.mode == "audit")
    if args.mode == "audit":
        print(json.dumps({"audit": audit}, sort_keys=True))
        return
    fixture = json.loads(Path(args.fixture_manifest).read_text(encoding="utf-8"))
    source = Path(args.fixture_manifest).parent / fixture["wal_base"]["name"]
    if run.sha256(source) != fixture["wal_base"]["sha256"]:
        raise RuntimeError("SDB WAL fixture differs from manifest")
    outcome = _read_scheduled(source, args.mutation, args.selection)
    print(json.dumps({"audit": audit, "sample": outcome}, sort_keys=True))


def _invoke(args: argparse.Namespace, arm: str, mode: str) -> dict[str, Any]:
    python = args.python_a if arm == "A" else args.python_b
    wheel = args.wheel_a if arm == "A" else args.wheel_b
    version = args.version_a if arm == "A" else args.version_b
    command = [
        python,
        "-I",
        str(Path(__file__).resolve()),
        "_worker",
        "--wheel",
        wheel,
        "--version",
        version,
        "--fixture-manifest",
        args.fixture_manifest,
        "--mutation",
        args.mutation,
        "--selection",
        args.selection,
        "--mode",
        mode,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    result["stderr"] = completed.stderr
    return result


def _make_fixture(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    generated = make_sdb_fixtures(destination / "sdb", rows=64)
    wal = generated["cases"]["wal_base"]
    _write_new(
        destination / "fixture-manifest.json",
        {
            "schema": 2,
            "generator": "benchmarks/io/format_fixtures.py:make_sdb_fixtures",
            "rows": 64,
            "wal_base": {**wal, "name": f"sdb/{wal['name']}"},
            "source_manifest_sha256": generated["manifest_sha256"],
        },
    )


def _capture(args: argparse.Namespace) -> None:
    if args.samples != 5:
        raise ValueError("SDB WAL baseline requires five samples per arm")
    fixture_manifest = Path(args.fixture_manifest)
    fixture = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    source = fixture_manifest.parent / fixture["wal_base"]["name"]
    if run.sha256(source) != fixture["wal_base"]["sha256"]:
        raise RuntimeError("SDB WAL fixture differs from manifest")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    audits = {arm: _invoke(args, arm, "audit")["audit"] for arm in ("A", "B")}
    if (
        audits["A"]["python"] != audits["B"]["python"]
        or audits["A"]["distributions"] != audits["B"]["distributions"]
    ):
        raise RuntimeError("Python or dependency environment differs between arms")
    _write_new(
        destination / "manifest.json",
        {
            "schema": 2,
            "status": "UNBASELINED",
            "lane": "B-F2-SDB-v2",
            "mutation": args.mutation,
            "selection": args.selection,
            "order": ORDER,
            "samples_per_arm": args.samples,
            "harness_digest": run.harness_digest(),
            "fixture_manifest_sha256": run.sha256(fixture_manifest),
            "fixture_sha256": fixture["wal_base"]["sha256"],
            "arms": {
                arm: {
                    **audits[arm],
                    "label": "B0" if arm == "A" else "B1",
                    "source_sha": args.source_sha_a
                    if arm == "A"
                    else args.source_sha_b,
                    "install_mode": "wheel-no-deps",
                }
                for arm in ("A", "B")
            },
        },
    )
    records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for index, arm in enumerate(ORDER):
        result = _invoke(args, arm, "correctness")
        if any(result["audit"][key] != audits[arm][key] for key in result["audit"]):
            raise RuntimeError("installed wheel identity changed during run")
        if not result["sample"]["events"]["triggered"]:
            raise RuntimeError("WAL mutation was not triggered at payload SELECT")
        if not result["sample"]["events"].get("writer_committed"):
            raise RuntimeError("WAL mutation did not commit")
        if result["sample"]["events"]["wal_mode"] != "wal":
            raise RuntimeError("mutation did not run in SQLite WAL mode")
        before_sql = result["sample"]["events"]["reader_sql_before_trigger"]
        if not any("usunits" in sql.casefold() for sql in before_sql):
            raise RuntimeError("mutation ran before source metadata validation")
        records[arm].append(result)
        _write_new(destination / f"sample-{index:02d}-{arm}.json", result)
    for arm in ("A", "B"):
        _write_new(destination / f"raw-{arm}.json", records[arm])

    def comparable(record: dict[str, Any]) -> dict[str, Any]:
        sample = record["sample"].copy()
        sample.pop("traceback", None)
        return sample

    a = comparable(records["A"][0])
    b = comparable(records["B"][0])
    _write_new(
        destination / "fingerprint.json",
        {
            "A": a,
            "B": b,
            "equal": a == b,
            "stable_A": all(comparable(record) == a for record in records["A"]),
            "stable_B": all(comparable(record) == b for record in records["B"]),
        },
    )
    print(destination)


def main() -> None:
    """Generate one deterministic fixture or capture a two-arm fault case."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fixtures = sub.add_parser("fixtures")
    fixtures.add_argument("destination")
    capture = sub.add_parser("capture")
    capture.add_argument("--fixture-manifest", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--mutation", choices=tuple(MUTATIONS), required=True)
    capture.add_argument("--selection", choices=SELECTIONS, required=True)
    capture.add_argument("--samples", type=int, default=5)
    for suffix in ("a", "b"):
        capture.add_argument(f"--python-{suffix}", required=True)
        capture.add_argument(f"--wheel-{suffix}", required=True)
        capture.add_argument(f"--version-{suffix}", required=True)
        capture.add_argument(f"--source-sha-{suffix}", required=True)
    worker = sub.add_parser("_worker")
    worker.add_argument("--wheel", required=True)
    worker.add_argument("--version", required=True)
    worker.add_argument("--fixture-manifest", required=True)
    worker.add_argument("--mutation", choices=tuple(MUTATIONS), required=True)
    worker.add_argument("--selection", choices=SELECTIONS, required=True)
    worker.add_argument("--mode", choices=("audit", "correctness"), required=True)
    args = parser.parse_args()
    if args.action == "fixtures":
        _make_fixture(Path(args.destination))
    elif args.action == "capture":
        _capture(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
