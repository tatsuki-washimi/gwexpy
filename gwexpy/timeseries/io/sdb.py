"""SDB reader for Davis Vantage Pro2 and WeeWX SQLite files."""

from __future__ import annotations

import re
import sqlite3
import warnings
from math import floor, isfinite
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from astropy.time import Time

from gwexpy.io.time_selection import (
    _normalize_bound,
    apply_time_selection,
    pop_time_selection,
)
from gwexpy.io.utils import _reject_timezone_reinterpretation

from .. import TimeSeries, TimeSeriesDict
from ._multi import expand_multi_source, read_multi_dict
from ._registration import register_timeseries_format

_SDB_WARNING_SCAN_CHUNK_ROWS = 256
_SQLITE_INT64_MIN = -(2**63)
_SQLITE_INT64_MAX = 2**63 - 1
_SDB_CANONICAL_INTEGER_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)")
_SDB_INTEGER_TEXT_FLOAT64_LIMIT = 2**53

# Unit extraction factors (Imperial to Metric)
UNIT_CONVERSION = {
    "barometer": ("hPa", 33.8639),  # inHg -> hPa
    "pressure": ("hPa", 33.8639),  # inHg -> hPa
    "altimeter": ("hPa", 33.8639),  # inHg -> hPa
    "inTemp": ("deg_C", lambda x: (x - 32) / 1.8),
    "outTemp": ("deg_C", lambda x: (x - 32) / 1.8),
    "dewpoint": ("deg_C", lambda x: (x - 32) / 1.8),
    "windchill": ("deg_C", lambda x: (x - 32) / 1.8),
    "heatindex": ("deg_C", lambda x: (x - 32) / 1.8),
    "extraTemp1": ("deg_C", lambda x: (x - 32) / 1.8),
    "extraTemp2": ("deg_C", lambda x: (x - 32) / 1.8),
    "extraTemp3": ("deg_C", lambda x: (x - 32) / 1.8),
    "soilTemp1": ("deg_C", lambda x: (x - 32) / 1.8),
    "rain": ("mm", 25.4),  # inch -> mm
    "rainRate": ("mm/h", 25.4),  # inch/h -> mm/h
    "windSpeed": ("m/s", 0.44704),  # mph -> m/s
    "windGust": ("m/s", 0.44704),  # mph -> m/s
    "inHumidity": ("%", 1.0),
    "outHumidity": ("%", 1.0),
    "radiation": ("W/m^2", 1.0),
    "UV": ("", 1.0),
}


def _quote_sqlite_identifier(identifier: str) -> str:
    """Return *identifier* as a safely quoted SQLite identifier."""
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("SQLite identifiers must be non-empty strings")
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _validate_us_units(conn: sqlite3.Connection, table: str) -> None:
    """Require WeeWX ``usUnits`` values to be the supported unit system."""
    table_identifier = _quote_sqlite_identifier(table)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_identifier})")  # nosec B608
    if "usUnits" not in {info[1] for info in cursor.fetchall()}:
        return

    cursor.execute(  # nosec B608
        f'SELECT "dateTime", "usUnits" FROM {table_identifier} ORDER BY "dateTime"'
    )
    for date_time, value in cursor:
        if value is None:
            raise ValueError(
                f"SDB usUnits validation failed at dateTime {date_time!r}: "
                "NULL is not allowed; expected integer 1."
            )
        if isinstance(value, (int, np.integer)):
            numeric_value = int(value)
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value) or not value.is_integer():
                raise ValueError(
                    f"SDB usUnits validation failed at dateTime {date_time!r}: "
                    f"non-integral value {value!r}; expected integer 1."
                )
            numeric_value = int(value)
        else:
            raise ValueError(
                f"SDB usUnits validation failed at dateTime {date_time!r}: "
                f"non-numeric value {value!r}; expected integer 1."
            )
        if numeric_value != 1:
            raise ValueError(
                f"SDB usUnits validation failed at dateTime {date_time!r}: "
                f"value {value!r} must be integer 1."
            )


def _exact_integer_upper_median(
    conn: sqlite3.Connection,
    query: str,
    delta_count: int,
    minimum: int,
    maximum: int,
) -> int:
    """Select the upper median of integer timestamp deltas in constant memory.

    This multi-pass value-domain search is used only when SQLite promoted a
    timestamp subtraction to REAL because the exact delta exceeds int64. The
    upper median matches ``sorted(deltas)[len(deltas) // 2]`` without retaining
    the deltas in Python.
    """
    target_index = delta_count // 2
    lower = minimum
    upper = maximum
    while lower < upper:
        midpoint = (lower + upper) // 2
        at_or_below = 0
        previous: int | None = None
        for row in conn.execute(query):
            value = int(row[0])
            if previous is not None and value - previous <= midpoint:
                at_or_below += 1
            previous = value
        if at_or_below > target_index:
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


def _unit_conversion_chunk_has_fp_events(column: str, values: np.ndarray) -> bool:
    """Return whether unit conversion raises a floating-point event in a chunk.

    Bounded reads still need to preserve the warnings and exceptions that the
    legacy full-array conversion would produce for rows outside the requested
    range. This preflight runs under strict NumPy error handling; a detected
    event makes the caller use the legacy full-payload route, where the
    caller's own ``np.seterr`` policy remains authoritative.
    """
    factor = UNIT_CONVERSION[column][1]
    try:
        with np.errstate(all="raise"):
            if callable(factor):
                factor(values)
            else:
                values * float(cast(float, factor))
    except FloatingPointError:
        return True
    return False


def _parse_sdb_integer_text(value: object) -> int | None:
    """Parse only canonical ASCII integer TEXT exactly representable in float64."""
    if (
        not isinstance(value, str)
        or value == "-0"
        or _SDB_CANONICAL_INTEGER_TEXT.fullmatch(value) is None
    ):
        # pandas.to_numeric preserves the IEEE-754 sign bit of this token.
        # Parsing it as Python int would lose that bit and change factor-1
        # unit columns when the bounded payload contains only this row.
        return None
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > 16:
        return None
    try:
        integer = int(value, 10)
    except ValueError:
        return None
    if abs(integer) >= _SDB_INTEGER_TEXT_FLOAT64_LIMIT:
        return None
    return integer


def _scan_sdb_timestamps(
    conn: sqlite3.Connection,
    table_identifier: str,
    order_clause: str,
    warning_columns: list[str] | None = None,
    unit_conversion_columns: list[str] | None = None,
) -> tuple[int | None, int, float, ValueError | None, dict[str, int], bool]:
    """Validate an integer SDB time grid without retaining its rows in Python.

    The previous implementation passed every timestamp through a Python list
    before validating cadence. This scan keeps only a few scalars. SQLite
    performs the median query only for an irregular grid so the error message
    and first mismatching index continue to match the existing validator. For
    bounded reads, text/blob values are checked in fixed-size chunks. Numeric
    values for selected unit-conversion columns are also converted in bounded
    float64 chunks to detect legacy NumPy warnings/errors without retaining the
    full payload in Python. Unit columns stored as TEXT are eligible only when
    every token is a canonical ASCII integer strictly inside float64's exact
    integer range; unqualified text or any BLOB delegates the full column to
    the legacy materializing path.

    Timestamp errors are returned instead of raised because the public reader
    emits selected-payload coercion warnings before it checks cadence.
    """
    warning_columns = [
        column for column in (warning_columns or []) if column != "dateTime"
    ]
    unit_conversion_columns = [
        column
        for column in (unit_conversion_columns or [])
        if column in warning_columns and column in UNIT_CONVERSION
    ]
    timestamp_query = (
        f'SELECT "dateTime" FROM {table_identifier} ORDER BY {order_clause}'
    )
    select_parts = ['"dateTime"']
    warning_column_indexes: dict[str, int] = {}
    unit_column_indexes: dict[str, int] = {}
    for column in warning_columns:
        quoted_column = _quote_sqlite_identifier(column)
        warning_column_indexes[column] = len(select_parts)
        select_parts.append(
            f"CASE WHEN typeof({quoted_column}) IN ('text', 'blob') "
            f"THEN {quoted_column} ELSE NULL END"
        )
        if column in unit_conversion_columns:
            unit_column_indexes[column] = len(select_parts)
            select_parts.append(
                f"CASE WHEN typeof({quoted_column}) IN ('integer', 'real') "
                f"THEN {quoted_column} ELSE NULL END"
            )
    query = (
        f"SELECT {', '.join(select_parts)} FROM {table_identifier} "
        f"ORDER BY {order_clause}"
    )
    first_timestamp: int | None = None
    row_count = 0
    previous: int | None = None
    first_delta: int | None = None
    uniform_grid = True
    ordering_error: ValueError | None = None
    timestamp_type_error: ValueError | None = None
    timestamp_type_error_rows = 0
    warning_counts = dict.fromkeys(warning_columns, 0)
    unit_conversion_safe = True

    def count_warning_chunk(rows: list[tuple[object, ...]]) -> None:
        nonlocal unit_conversion_safe
        for column in warning_columns:
            # Numeric storage is returned in a separate projection. The text
            # projection is NULL for integer/real/null storage.
            warning_index = warning_column_indexes[column]
            raw_values = [
                row[warning_index] for row in rows if row[warning_index] is not None
            ]
            if column in unit_column_indexes:
                if not unit_conversion_safe:
                    continue
                unit_index = unit_column_indexes[column]
                converted_values: list[float | int] = []
                for row in rows:
                    raw_value = row[warning_index]
                    if raw_value is not None:
                        integer = _parse_sdb_integer_text(raw_value)
                        if integer is None:
                            # The full route owns pandas coercion, warning
                            # counts, unit conversion, and their ordering.
                            unit_conversion_safe = False
                            break
                        converted_values.append(integer)
                    else:
                        numeric_value = row[unit_index]
                        if numeric_value is None:
                            converted_values.append(np.nan)
                        elif isinstance(
                            numeric_value, (int, float, np.integer, np.floating)
                        ):
                            converted_values.append(float(numeric_value))
                        else:
                            unit_conversion_safe = False
                            break
                if not unit_conversion_safe:
                    continue
                data = np.asarray(converted_values, dtype=np.float64)
                if _unit_conversion_chunk_has_fp_events(column, data):
                    unit_conversion_safe = False
                continue

            values = pd.Series(raw_values)
            if not values.empty:
                coerced = pd.to_numeric(values, errors="coerce")
                warning_counts[column] += int((coerced.isna() & values.notna()).sum())

    cursor = conn.execute(query)
    warning_batch: list[tuple[object, ...]] = []
    for index, row in enumerate(cursor):
        row_count = index + 1
        value = row[0]
        if timestamp_type_error is None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                timestamp_type_error = ValueError(
                    f"SDB dateTime at index {index} must contain integer Unix seconds"
                )
                timestamp_type_error_rows = index + 1
                first_timestamp = None
                if not warning_columns:
                    return (
                        None,
                        index + 1,
                        1.0,
                        timestamp_type_error,
                        warning_counts,
                        unit_conversion_safe,
                    )
            elif index == 0:
                first_timestamp = int(value)
            else:
                assert previous is not None
                delta = int(value) - previous
                if delta == 0 and ordering_error is None:
                    ordering_error = ValueError(
                        f"SDB duplicate timestamp at index {index}"
                    )
                elif delta < 0 and ordering_error is None:
                    ordering_error = ValueError(
                        f"SDB backward timestamp at index {index}"
                    )
                if first_delta is None:
                    first_delta = delta
                elif delta != first_delta:
                    uniform_grid = False
            if isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                previous = int(value)

        if warning_columns:
            warning_batch.append(row)
            if len(warning_batch) == _SDB_WARNING_SCAN_CHUNK_ROWS:
                count_warning_chunk(warning_batch)
                warning_batch.clear()

    if warning_batch:
        count_warning_chunk(warning_batch)

    if timestamp_type_error is not None:
        return (
            None,
            timestamp_type_error_rows,
            1.0,
            timestamp_type_error,
            warning_counts,
            unit_conversion_safe,
        )

    if ordering_error is not None:
        return (
            first_timestamp,
            row_count,
            1.0,
            ordering_error,
            warning_counts,
            unit_conversion_safe,
        )
    if row_count < 2:
        return (
            first_timestamp,
            row_count,
            1.0,
            None,
            warning_counts,
            unit_conversion_safe,
        )

    if uniform_grid:
        assert first_delta is not None
        return (
            first_timestamp,
            row_count,
            float(first_delta),
            None,
            warning_counts,
            unit_conversion_safe,
        )

    delta_count = row_count - 1
    delta_query = (
        "WITH ordered AS ("
        f'SELECT "dateTime" - LAG("dateTime") OVER (ORDER BY {order_clause}) '
        f"AS delta FROM {table_identifier}"
        ") "
    )
    overflow_query = (
        delta_query + "SELECT 1 FROM ordered WHERE delta IS NOT NULL "
        "AND typeof(delta) != 'integer' LIMIT 1"
    )
    if conn.execute(overflow_query).fetchone() is not None:
        # Duplicate/backward deltas were returned above, so each remaining
        # positive int64 timestamp delta lies within this exact domain.
        cadence = _exact_integer_upper_median(
            conn,
            timestamp_query,
            delta_count,
            1,
            2**64 - 1,
        )
    else:
        median_query = (
            delta_query + "SELECT delta FROM ordered WHERE delta IS NOT NULL "
            "ORDER BY delta LIMIT 1 OFFSET ?"
        )
        median_row = conn.execute(median_query, (delta_count // 2,)).fetchone()
        if median_row is None:
            raise RuntimeError("SDB timestamp median query returned no cadence")
        cadence = int(median_row[0])

    previous = None
    cursor = conn.execute(timestamp_query)
    for index, row in enumerate(cursor):
        value = int(row[0])
        if previous is not None and value - previous != cadence:
            delta = value - previous
            return (
                first_timestamp,
                row_count,
                float(cadence),
                ValueError(
                    f"SDB timestamp gap at index {index}: "
                    f"expected cadence {cadence}, got {delta}"
                ),
                warning_counts,
                unit_conversion_safe,
            )
        previous = value
    raise RuntimeError("SDB irregular timestamp grid had no mismatching cadence")


def _sdb_crop_index(
    bound_gps: float,
    t0_gps: float,
    source_dt: float,
) -> int:
    """Match fresh regular ``TimeSeries.crop`` index calculation.

    A newly constructed GWpy ``Series`` has no materialized ``_xindex`` and
    uses ``floor((xtype(bound) - x0) / dx)``.  In particular, this differs
    from ``searchsorted(left)`` for half-sample and one-ULP bounds.
    """
    x0 = np.float64(t0_gps)
    axis_type = type(x0)
    axis_bound = axis_type(bound_gps)
    sample_rate = 1.0 / source_dt
    dx = axis_type(1.0 / sample_rate)
    return floor((axis_bound - x0) / dx)


def _sdb_payload_window(
    first_timestamp: int,
    row_count: int,
    source_dt: float,
    t0_gps: float,
    start: object | None,
    end: object | None,
) -> tuple[int, int] | None:
    """Return the half-open Unix-second payload range for a positive window.

    The eligible route mirrors ``apply_time_selection``'s span clamp followed
    by fresh ``Series.crop`` floor indexing. Empty, disjoint, reversed, and
    non-finite selectors use the full-payload route so GWpy's slice epoch and
    error behavior stay authoritative.
    """
    if start is None and end is None:
        return None
    if (
        row_count <= 0
        or source_dt <= 0
        or not source_dt.is_integer()
        or abs(first_timestamp) >= 2**53
        or source_dt >= 2**53
    ):
        return None

    try:
        start_gps = _normalize_bound(start)
        end_gps = _normalize_bound(end)
    except (TypeError, ValueError, OverflowError):
        # Preserve the old warning/error order for invalid time selectors.
        return None

    try:
        if any(
            bound is not None and not isfinite(bound) for bound in (start_gps, end_gps)
        ):
            return None
        if start_gps is not None and end_gps is not None and start_gps > end_gps:
            return None

        # TimeSeries.xspan is [x0, x0 + row_count * dx).  Match the
        # apply_time_selection clamp order before evaluating crop indexes.
        x0 = np.float64(t0_gps)
        axis_type = type(x0)
        sample_rate = 1.0 / source_dt
        dx = axis_type(1.0 / sample_rate)
        if dx != source_dt:
            # SQL rows advance by the exact integer timestamp cadence, while
            # GWpy builds its axis from the reciprocal sample rate. If that
            # round-trip changes dx, rebasing a selected SQL subset can shift
            # its epoch relative to a fresh full-read crop.
            return None
        x1 = x0 + row_count * dx
        lo = float(x0)
        hi = float(x1)
        if end_gps is not None and end_gps <= lo:
            return None
        if start_gps is not None and start_gps >= hi:
            return None
        if start_gps is not None and start_gps <= lo:
            start_gps = None
        if end_gps is not None and end_gps >= hi:
            end_gps = None

        first_index = (
            0 if start_gps is None else _sdb_crop_index(start_gps, t0_gps, source_dt)
        )
        end_index = (
            row_count
            if end_gps is None
            else _sdb_crop_index(end_gps, t0_gps, source_dt)
        )
        if first_index >= end_index:
            # Empty crops retain a selector-dependent t0 when made by GWpy;
            # rebuilding an empty array at the source epoch is not equivalent.
            return None

        cadence_seconds = int(source_dt)
        lower = first_timestamp + first_index * cadence_seconds
        upper = first_timestamp + end_index * cadence_seconds
        if not (
            _SQLITE_INT64_MIN <= lower <= _SQLITE_INT64_MAX
            and _SQLITE_INT64_MIN <= upper <= _SQLITE_INT64_MAX
        ):
            # SQLite bind parameters use signed int64. The exclusive upper
            # endpoint can be one cadence past a valid final stored timestamp.
            return None
        return lower, upper
    except (TypeError, ValueError, OverflowError):
        # Let apply_time_selection raise after payload warnings as it did before
        # window pushdown was added.
        return None


def read_timeseriesdict_sdb(
    source: str | Path, table="archive", columns=None, **kwargs
):
    """Read SDB (SQLite) file into TimeSeriesDict.

    Parameters
    ----------
    source : str, Path, or list of str/Path
        Path to SQLite database file, or a list of paths.  When a list
        is given, columns found in several databases are concatenated
        along the time axis and columns unique to one database are
        merged in.
    table : str, optional
        Table name to read from, default 'archive'.
    columns : list, optional
        List of column names to read. If None, reads all columns found in UNIT_CONVERSION + dateTime.
        ``usUnits`` is validated separately and is never returned as a series.
        If the archive contains that column, every value must be integer ``1``;
        archives without it retain the legacy US customary unit assumption.
    **kwargs
        Additional compatibility arguments accepted and ignored.  ``start`` and
        ``end`` are the exception: they used to be ignored here too, so a
        bounded read quietly returned every row in the table (issue #611).  They
        are now honoured by cropping the assembled result.

    """
    start, end = pop_time_selection(kwargs)
    timezone = kwargs.pop("timezone", None)
    kwargs.pop("epoch", None)
    _reject_timezone_reinterpretation("sdb", timezone, None)

    multi = expand_multi_source(source)
    if multi is not None:
        return apply_time_selection(
            read_multi_dict(
                read_timeseriesdict_sdb,
                multi,
                "sdb",
                table=table,
                columns=columns,
                **kwargs,
            ),
            start,
            end,
        )

    # gwpy's registry may pass an already-open file object for explicit
    # ``.read(..., format="sdb")`` calls; sqlite3 needs the underlying path.
    if not isinstance(source, (str, Path)) and hasattr(source, "name"):
        source = source.name

    # Open SQLite connection (Python 3.4+ accepts Path objects)
    conn = sqlite3.connect(source)

    timestamp_error: ValueError | None = None
    first_timestamp: int | None = None
    source_rows = 0
    source_dt = 1.0
    payload_window: tuple[int, int] | None = None
    warning_counts: dict[str, int] = {}
    try:
        table_identifier = _quote_sqlite_identifier(table)
        # Pin a WAL read snapshot before looking at table metadata.  Every
        # validation scan and the eventual payload query then observes the
        # same schema and rows, even if a writer commits between them.
        conn.execute("BEGIN")
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        _validate_us_units(conn, table)

        # Determine columns to query
        known_column_names: set[str] | None = None
        if columns is None:
            # Check available columns in the table using PRAGMA
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_identifier})")
            table_cols = [info[1] for info in cursor.fetchall()]
            known_column_names = {str(column) for column in table_cols}

            # Filter columns that we know how to convert + dynamic others?
            # For now, stick to known weather columns
            target_cols = [
                c for c in table_cols if c in UNIT_CONVERSION or c == "dateTime"
            ]
        else:
            target_cols = [c for c in columns if c != "usUnits"]
            if "dateTime" not in target_cols:
                target_cols.append("dateTime")

        # A rowid table is traversed in SQLite rowid/B-tree order.  This is a
        # deterministic storage order, not insertion chronology when an
        # INTEGER PRIMARY KEY aliases rowid.  Validate that order so a record
        # cannot be silently repaired by a timestamp sort. Determine WITHOUT
        # ROWID from schema metadata rather than probing ``rowid``, which a
        # declared column can shadow.
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_list({table_identifier})")  # nosec B608
        table_records = [
            info
            for info in cursor.fetchall()
            if len(info) >= 5
            and str(info[1]).casefold() == table.casefold()
            and info[2] == "table"
        ]
        if len(table_records) != 1:
            raise ValueError(
                "SDB source row order cannot be established from table metadata"
            )

        if bool(table_records[0][4]):
            cursor.execute(f"PRAGMA index_list({table_identifier})")  # nosec B608
            primary_key_indexes = [
                str(info[1]) for info in cursor.fetchall() if info[3] == "pk"
            ]
            if len(primary_key_indexes) != 1:
                raise ValueError(
                    "SDB source row order cannot be established for a "
                    "WITHOUT ROWID table without exactly one declared primary key"
                )
            primary_key_identifier = _quote_sqlite_identifier(primary_key_indexes[0])
            cursor.execute(  # nosec B608
                f"PRAGMA index_xinfo({primary_key_identifier})"
            )
            primary_key_parts = sorted(
                (int(info[0]), info) for info in cursor.fetchall() if int(info[5]) == 1
            )
            order_columns = []
            for _, info in primary_key_parts:
                column_name = info[2]
                if column_name is None or int(info[1]) < 0:
                    raise ValueError(
                        "SDB source row order cannot be established from an "
                        "expression-based primary key"
                    )
                order_part = _quote_sqlite_identifier(str(column_name))
                collation = info[4]
                if collation:
                    order_part += " COLLATE " + _quote_sqlite_identifier(str(collation))
                order_part += " DESC" if int(info[3]) else " ASC"
                order_columns.append(order_part)
            if not order_columns:
                raise ValueError(
                    "SDB source row order cannot be established from the "
                    "declared primary key"
                )
        else:
            cursor.execute(f"PRAGMA table_info({table_identifier})")  # nosec B608
            column_names = [str(info[1]) for info in cursor.fetchall()]
            declared_columns = {column.casefold() for column in column_names}
            known_column_names = set(column_names)
            hidden_rowid = next(
                (
                    alias
                    for alias in ("rowid", "_rowid_", "oid")
                    if alias.casefold() not in declared_columns
                ),
                None,
            )
            if hidden_rowid is None:
                raise ValueError(
                    "SDB source row order cannot be established because all "
                    "SQLite rowid aliases are shadowed"
                )
            order_columns = [_quote_sqlite_identifier(hidden_rowid)]

        order_clause = ", ".join(order_columns)

        target_names = [column for column in target_cols if isinstance(column, str)]
        normalized_target_names = [column.casefold() for column in target_names]
        window_scan_eligible = (
            (start is not None or end is not None)
            and len(target_names) == len(target_cols)
            and len(set(normalized_target_names)) == len(target_names)
        )
        if window_scan_eligible and known_column_names is None:
            # WITHOUT ROWID sources do not need table_info to establish order.
            # Read it only here so fast-path SQL never treats an unknown quoted
            # selector as a SQLite double-quoted string literal.
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_identifier})")  # nosec B608
            known_column_names = {str(info[1]) for info in cursor.fetchall()}
        if window_scan_eligible:
            assert known_column_names is not None
            window_scan_eligible = all(
                name in known_column_names for name in target_names
            )

        scan_warning_columns = (
            [col for col in target_cols if col != "dateTime"]
            if window_scan_eligible
            else []
        )
        scanned_warning_counts: dict[str, int] = {}
        scanned_unit_conversion_safe = True
        if "dateTime" in target_cols:
            (
                first_timestamp,
                source_rows,
                source_dt,
                timestamp_error,
                scanned_warning_counts,
                scanned_unit_conversion_safe,
            ) = _scan_sdb_timestamps(
                conn,
                table_identifier,
                order_clause,
                warning_columns=scan_warning_columns,
                unit_conversion_columns=[
                    column
                    for column in scan_warning_columns
                    if column in UNIT_CONVERSION
                ],
            )
        else:
            # Preserve the legacy warning-before-error order when the default
            # column selection finds payload fields but no dateTime column.
            timestamp_error = ValueError(
                "Table must contain 'dateTime' column for time series conversion."
            )

        if (
            window_scan_eligible
            and timestamp_error is None
            and first_timestamp is not None
            and scanned_unit_conversion_safe
        ):
            t0_gps = float(Time(float(first_timestamp), format="unix").gps)
            payload_window = _sdb_payload_window(
                first_timestamp,
                source_rows,
                source_dt,
                t0_gps,
                start,
                end,
            )
            if payload_window is not None:
                warning_counts = scanned_warning_counts
        # On timestamp errors or selectors that cannot use SQL range pushdown,
        # keep the legacy full-payload fallback. Its DataFrame coercion owns
        # warning emission, so any warning counts gathered during the combined
        # scan are intentionally discarded.

        col_str = ", ".join(_quote_sqlite_identifier(c) for c in target_cols)
        query = f"SELECT {col_str} FROM {table_identifier}"
        params: tuple[object, ...] = ()
        if payload_window is not None:
            query += ' WHERE "dateTime" >= ? AND "dateTime" < ?'
            params = payload_window
        query += f" ORDER BY {order_clause}"

        # Only the requested payload window enters a DataFrame.  Validation
        # above uses cursor iteration and keeps constant Python-side memory.
        df = pd.read_sql_query(query, conn, params=params)
        conn.commit()

    finally:
        conn.close()

    if df.empty and source_rows == 0:
        return TimeSeriesDict()

    if df.empty and payload_window is not None:
        # A zero-row payload window still needs to return the same keys and
        # epoch as cropping a full read. Count warnings against the whole
        # source above, then reconstruct only empty arrays here.
        for col in target_cols:
            if col == "dateTime":
                continue
            warning_count = warning_counts.get(col, 0)
            if warning_count:
                warnings.warn(
                    f"SDB column '{col}': {warning_count} non-numeric value(s) "
                    "could not be parsed and were set to NaN.",
                    UserWarning,
                    stacklevel=3,
                )
        if timestamp_error is not None:
            raise timestamp_error
        assert first_timestamp is not None
        assert source_dt > 0
        empty_t0_gps = t0_gps
        sample_rate = 1.0 / source_dt
        tsd = TimeSeriesDict()
        for col in target_cols:
            if col == "dateTime":
                continue
            unit = UNIT_CONVERSION[col][0] if col in UNIT_CONVERSION else ""
            tsd[col] = TimeSeries(
                np.empty(0, dtype=np.float64),
                t0=empty_t0_gps,
                sample_rate=sample_rate,
                name=col,
                unit=unit,
            )
        return apply_time_selection(tsd, start, end)

    # Coerce numeric columns to handle potential NULLs -> NaN
    # We expect all columns except potentially descriptive ones (none here) to be numeric
    for col in df.columns:
        if col != "dateTime":
            coerced = pd.to_numeric(df[col], errors="coerce")
            # Count values that were non-null before but became NaN: these are
            # genuinely unparseable entries silently dropped by errors="coerce".
            lost = int((coerced.isna() & df[col].notna()).sum())
            warning_count = (
                warning_counts.get(col, lost) if payload_window is not None else lost
            )
            if warning_count:
                warnings.warn(
                    f"SDB column '{col}': {warning_count} non-numeric value(s) could not "
                    f"be parsed and were set to NaN.",
                    UserWarning,
                    stacklevel=3,
                )
            df[col] = coerced

    # Check if dateTime is present
    if "dateTime" not in df.columns:
        raise ValueError(
            "Table must contain 'dateTime' column for time series conversion."
        )

    if timestamp_error is not None:
        raise timestamp_error

    # Convert dateTime to GPS time (it's usually UNIX timestamp)
    # TimeSeries expects t0 in GPS. Unix to GPS is roughly +18s (leap seconds).
    # gwpy.time.to_gps handles datetime objects.
    # Convert first timestamp
    payload_first_timestamp = int(df["dateTime"].iloc[0])
    assert first_timestamp is not None
    payload_offset = (payload_first_timestamp - first_timestamp) / source_dt
    payload_t0_gps = (
        float(Time(float(first_timestamp), format="unix").gps)
        + payload_offset * source_dt
    )

    sample_rate = 1.0 / source_dt

    # Convert to TimeSeriesDict
    tsd = TimeSeriesDict()

    # Use astropy Time for accurate conversion if needed, but simple offset is faster
    # Unix 0 = 1970-01-01 00:00:00 UTC = GPS 315964819 (wait, 315964819 is with leap seconds)
    # Correct way: Time(unix_val, format='unix').gps
    t0_gps = payload_t0_gps

    for col in df.columns:
        if col == "dateTime":
            continue

        data = np.asarray(df[col].to_numpy(), dtype=np.float64)
        unit = ""

        # Apply conversion
        if col in UNIT_CONVERSION:
            u_name, factor = UNIT_CONVERSION[col]
            unit = u_name
            if callable(factor):
                data = factor(data)
            else:
                data = data * float(cast(float, factor))

        ts = TimeSeries(data, t0=t0_gps, sample_rate=sample_rate, name=col, unit=unit)
        tsd[col] = ts

    if payload_window is not None:
        return tsd
    return apply_time_selection(tsd, start, end)


def read_timeseries_sdb(source, **kwargs):
    """Read an SDB file as a ``TimeSeries``.

    If multiple columns, returns the first one (excluding dateTime).
    """
    tsd = read_timeseriesdict_sdb(source, **kwargs)
    if not tsd:
        raise ValueError("No time series data found in sdb file")
    return tsd[next(iter(tsd.keys()))]


# -- Registration

register_timeseries_format(
    "sdb",
    reader_dict=read_timeseriesdict_sdb,
    reader_single=read_timeseries_sdb,
    extension="sdb",
)
