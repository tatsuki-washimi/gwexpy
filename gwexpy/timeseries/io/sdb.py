"""SDB reader for Davis Vantage Pro2 and WeeWX SQLite files."""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from astropy.time import Time

from gwexpy.io.time_selection import apply_time_selection, pop_time_selection
from gwexpy.io.utils import (
    _reject_timezone_reinterpretation,
    _validate_regular_timestamps,
)

from .. import TimeSeries, TimeSeriesDict
from ._multi import expand_multi_source, read_multi_dict
from ._registration import register_timeseries_format

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

    try:
        table_identifier = _quote_sqlite_identifier(table)
        _validate_us_units(conn, table)

        # Determine columns to query
        if columns is None:
            # Check available columns in the table using PRAGMA
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_identifier})")
            table_cols = [info[1] for info in cursor.fetchall()]

            # Filter columns that we know how to convert + dynamic others?
            # For now, stick to known weather columns
            target_cols = [
                c for c in table_cols if c in UNIT_CONVERSION or c == "dateTime"
            ]
        else:
            target_cols = [c for c in columns if c != "usUnits"]
            if "dateTime" not in target_cols:
                target_cols.append("dateTime")

        # A rowid table preserves archive insertion order.  Validate that
        # order so an out-of-order record cannot be silently repaired by a
        # timestamp sort.  WITHOUT ROWID tables are stored in declared
        # primary-key order, which PRAGMA reports using one-based PK ordinals.
        cursor = conn.cursor()
        try:
            cursor.execute(  # nosec B608
                f"SELECT rowid FROM {table_identifier} LIMIT 0"
            )
        except sqlite3.OperationalError:
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
            order_columns = ["rowid"]

        col_str = ", ".join(_quote_sqlite_identifier(c) for c in target_cols)
        order_clause = ", ".join(order_columns)
        query = (  # nosec B608
            f"SELECT {col_str} FROM {table_identifier} ORDER BY {order_clause}"
        )

        # Use pandas for easy reading using the connection context
        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    if df.empty:
        return TimeSeriesDict()

    # Coerce numeric columns to handle potential NULLs -> NaN
    # We expect all columns except potentially descriptive ones (none here) to be numeric
    for col in df.columns:
        if col != "dateTime":
            coerced = pd.to_numeric(df[col], errors="coerce")
            # Count values that were non-null before but became NaN: these are
            # genuinely unparseable entries silently dropped by errors="coerce".
            lost = int((coerced.isna() & df[col].notna()).sum())
            if lost:
                warnings.warn(
                    f"SDB column '{col}': {lost} non-numeric value(s) could not "
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

    # Verify the native integer Unix-second grid before any conversion to
    # float.  A source gap cannot safely be repaired by inferring a rate.
    raw_time_values = df["dateTime"].to_list()
    for index, value in enumerate(raw_time_values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(
                f"SDB dateTime at index {index} must contain integer Unix seconds"
            )
    source_dt = _validate_regular_timestamps(raw_time_values, source="SDB")

    # Convert dateTime to GPS time (it's usually UNIX timestamp)
    # TimeSeries expects t0 in GPS. Unix to GPS is roughly +18s (leap seconds).
    # gwpy.time.to_gps handles datetime objects.
    # Convert first timestamp
    time_values = np.asarray(df["dateTime"].to_numpy(), dtype=np.float64)
    t_start_unix = time_values[0]

    sample_rate = 1.0 / source_dt

    # Convert to TimeSeriesDict
    tsd = TimeSeriesDict()

    # Use astropy Time for accurate conversion if needed, but simple offset is faster
    # Unix 0 = 1970-01-01 00:00:00 UTC = GPS 315964819 (wait, 315964819 is with leap seconds)
    # Correct way: Time(unix_val, format='unix').gps
    t0_gps = Time(t_start_unix, format="unix").gps

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
