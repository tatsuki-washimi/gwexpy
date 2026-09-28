"""SDB snapshot behavior for commits between validation and payload reads."""

from __future__ import annotations

import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from gwexpy.io.time_selection import apply_time_selection
from gwexpy.timeseries import TimeSeriesDict
from gwexpy.timeseries.io import sdb


def _create_wal_database(path, *, include_units: bool, sample_dt: int = 300) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        unit_column = ", usUnits INTEGER" if include_units else ""
        insert_sql = (
            "INSERT INTO archive VALUES (?, ?, ?, ?)"
            if include_units
            else "INSERT INTO archive VALUES (?, ?, ?)"
        )
        connection.execute(
            "CREATE TABLE archive ("
            "dateTime INTEGER, outTemp REAL, "
            f"outHumidity REAL{unit_column})"
        )
        connection.executemany(
            insert_sql,
            [
                (1_700_000_000 + index * sample_dt, 60.0 + index, 40.0 + index)
                + ((1,) if include_units else ())
                for index in range(64)
            ],
        )


_CASES = [
    pytest.param("selected_value", ["outTemp"], id="selected-value-selected"),
    pytest.param("selected_value", None, id="selected-value-all"),
    pytest.param("unselected_value", ["outTemp"], id="unselected-value-selected"),
    pytest.param("unselected_value", None, id="unselected-value-all"),
    pytest.param("selected_malformed", ["outTemp"], id="selected-malformed-selected"),
    pytest.param("selected_malformed", None, id="selected-malformed-all"),
    pytest.param(
        "unselected_malformed", ["outTemp"], id="unselected-malformed-selected"
    ),
    pytest.param("unselected_malformed", None, id="unselected-malformed-all"),
    pytest.param("timestamp", ["outTemp"], id="timestamp-selected"),
    pytest.param("timestamp", None, id="timestamp-all"),
    pytest.param("usUnits", ["outTemp"], id="usunits-selected"),
    pytest.param("usUnits", None, id="usunits-all"),
    pytest.param("schema", ["outTemp"], id="schema-selected"),
    pytest.param("schema", None, id="schema-all"),
]


def _commit_mutation(path, mutation: str) -> None:
    with sqlite3.connect(path) as writer:
        if mutation == "selected_value":
            writer.execute("UPDATE archive SET outTemp = 99 WHERE rowid = 1")
        elif mutation == "unselected_value":
            writer.execute("UPDATE archive SET outHumidity = 99 WHERE rowid = 1")
        elif mutation == "selected_malformed":
            writer.execute(
                "UPDATE archive SET outTemp = 'not-a-number' WHERE rowid = 1"
            )
        elif mutation == "unselected_malformed":
            writer.execute(
                "UPDATE archive SET outHumidity = 'not-a-number' WHERE rowid = 1"
            )
        elif mutation == "timestamp":
            writer.execute("UPDATE archive SET dateTime = dateTime + 1 WHERE rowid = 1")
        elif mutation == "usUnits":
            writer.execute("UPDATE archive SET usUnits = 2 WHERE rowid = 1")
        elif mutation == "schema":
            writer.execute(
                "ALTER TABLE archive RENAME COLUMN outTemp TO outTempRenamed"
            )
        else:  # pragma: no cover - protects future matrix additions
            raise AssertionError(f"unknown SDB mutation {mutation!r}")


@pytest.mark.parametrize(("mutation", "columns"), _CASES)
def test_sdb_validation_and_payload_use_one_wal_snapshot(
    tmp_path, monkeypatch, mutation: str, columns: list[str] | None
) -> None:
    """Concurrent commits cannot mix validation metadata and payload rows."""
    path = tmp_path / "snapshot.sdb"
    _create_wal_database(path, include_units=mutation == "usUnits")

    read_sql_query = pd.read_sql_query
    committed = False

    def commit_before_payload(query, connection, *args, **kwargs):
        nonlocal committed
        if not committed:
            _commit_mutation(path, mutation)
            committed = True
        return read_sql_query(query, connection, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "read_sql_query", commit_before_payload)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sdb.read_timeseriesdict_sdb(path, columns=columns)

    assert committed, "the scheduled WAL writer did not commit before payload fetch"
    assert not caught
    np.testing.assert_allclose(
        result["outTemp"].value[0], (60.0 - 32.0) / 1.8, rtol=1e-6
    )
    assert len(result["outTemp"]) == 64
    if columns is None:
        assert result["outHumidity"].value[0] == 40.0


def test_sdb_bounded_read_fetches_only_window_payload_and_keeps_warning_count(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "bounded.sdb"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, payload)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (
                    1_700_000_000 + index * 300,
                    "not-a-number" if index == 0 else 40.0 + index,
                )
                for index in range(64)
            ],
        )

    with pytest.warns(UserWarning, match="1 non-numeric value"):
        full = sdb.read_timeseriesdict_sdb(path, columns=["payload"])
    start = float(full["payload"].times[32].value)
    end = float(full["payload"].times[34].value)
    read_sql_query = pd.read_sql_query
    queries: list[str] = []
    payload_rows: list[int] = []

    def record_payload(query, connection, *args, **kwargs):
        queries.append(query)
        frame = read_sql_query(query, connection, *args, **kwargs)
        payload_rows.append(len(frame))
        return frame

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload)
    with pytest.warns(UserWarning, match="1 non-numeric value"):
        result = sdb.read_timeseriesdict_sdb(
            path, columns=["payload"], start=start, end=end
        )

    assert len(queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' in queries[0]
    assert payload_rows == [2]
    expected = full["payload"].crop(start, end)
    np.testing.assert_array_equal(result["payload"].value, expected.value)


@pytest.mark.parametrize(
    "selector",
    [
        "exact",
        "start_ulp_below",
        "start_ulp_above",
        "end_ulp_below",
        "end_ulp_above",
        "half_sample",
        "start_only",
        "end_only",
        "before_source",
        "after_source",
        "zero_width",
        "empty_inside",
    ],
)
@pytest.mark.parametrize("sample_dt", [300, 7])
def test_sdb_window_matches_crop_and_fetches_only_selected_rows(
    tmp_path, monkeypatch, selector: str, sample_dt: int
) -> None:
    path = tmp_path / "exact-window.sdb"
    _create_wal_database(path, include_units=False, sample_dt=sample_dt)
    full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    series = full["outTemp"]
    assert not hasattr(series, "_xindex")
    t0 = float(series.t0.value)
    dt = float(series.dt.value)
    start_exact, end_exact = t0 + 20 * dt, t0 + 24 * dt
    selectors = {
        "exact": (start_exact, end_exact),
        "start_ulp_below": (np.nextafter(start_exact, -np.inf), end_exact),
        "start_ulp_above": (np.nextafter(start_exact, np.inf), end_exact),
        "end_ulp_below": (start_exact, np.nextafter(end_exact, -np.inf)),
        "end_ulp_above": (start_exact, np.nextafter(end_exact, np.inf)),
        "half_sample": (start_exact + dt / 2, end_exact + dt / 2),
        "start_only": (start_exact, None),
        "end_only": (None, end_exact),
        "before_source": (t0 - 2 * dt, t0 - dt),
        "after_source": (t0 + 65 * dt, t0 + 66 * dt),
        "zero_width": (start_exact, start_exact),
        "empty_inside": (start_exact + dt / 2, start_exact + dt / 2),
    }
    sql_pushdown_expected = selector not in {
        "before_source",
        "after_source",
        "zero_width",
        "empty_inside",
    }
    start, end = selectors[selector]
    expected = apply_time_selection(full, start, end)["outTemp"]

    read_sql_query = pd.read_sql_query
    queries: list[str] = []
    payload_rows: list[int] = []

    def record_payload(query, connection, *args, **kwargs):
        queries.append(query)
        frame = read_sql_query(query, connection, *args, **kwargs)
        payload_rows.append(len(frame))
        return frame

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload)
    result = sdb.read_timeseriesdict_sdb(
        path,
        columns=["outTemp"],
        start=start,
        end=end,
    )["outTemp"]

    assert len(queries) == 1
    if sql_pushdown_expected:
        assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' in queries[0]
        assert payload_rows == [len(expected)]
    else:
        assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in queries[0]
        assert payload_rows == [64]
    np.testing.assert_array_equal(result.value, expected.value)
    np.testing.assert_array_equal(result.times.value, expected.times.value)
    assert result.t0 == expected.t0


@pytest.mark.parametrize(
    ("bound_name", "bound_value"),
    [
        pytest.param("start", np.nan, id="nan-start"),
        pytest.param("start", np.inf, id="posinf-start"),
        pytest.param("start", -np.inf, id="neginf-start"),
        pytest.param("end", np.nan, id="nan-end"),
        pytest.param("end", np.inf, id="posinf-end"),
        pytest.param("end", -np.inf, id="neginf-end"),
    ],
)
def test_sdb_nonfinite_window_uses_full_payload_route_and_matches_crop(
    tmp_path, monkeypatch, bound_name: str, bound_value: float
) -> None:
    path = tmp_path / "nonfinite-window.sdb"
    _create_wal_database(path, include_units=False)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE archive SET outTemp = 'not-a-number' WHERE rowid = 1"
        )

    with pytest.warns(UserWarning, match="1 non-numeric value"):
        full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    series = full["outTemp"]
    assert not hasattr(series, "_xindex")
    t0 = float(series.t0.value)
    dt = float(series.dt.value)
    selectors = {"start": t0 + 20 * dt, "end": t0 + 24 * dt}
    selectors[bound_name] = bound_value
    start, end = selectors["start"], selectors["end"]

    def capture(call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = call()
            except Exception as error:  # compare the public error signature too
                return (
                    "error",
                    type(error),
                    str(error),
                    [(item.category, str(item.message)) for item in caught],
                )
        return (
            "return",
            result,
            [(item.category, str(item.message)) for item in caught],
        )

    expected = capture(
        lambda: apply_time_selection(
            sdb.read_timeseriesdict_sdb(path, columns=["outTemp"]), start, end
        )["outTemp"]
    )

    read_sql_query = pd.read_sql_query
    queries: list[str] = []
    payload_rows: list[int] = []

    def record_payload(query, connection, *args, **kwargs):
        queries.append(query)
        frame = read_sql_query(query, connection, *args, **kwargs)
        payload_rows.append(len(frame))
        return frame

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload)
    actual = capture(
        lambda: sdb.read_timeseriesdict_sdb(
            path,
            columns=["outTemp"],
            start=start,
            end=end,
        )["outTemp"]
    )

    assert expected[0] == actual[0]
    if expected[0] == "error":
        assert expected[1:] == actual[1:]
    else:
        expected_series, expected_warnings = expected[1:]
        actual_series, actual_warnings = actual[1:]
        assert expected_warnings == actual_warnings
        np.testing.assert_array_equal(actual_series.value, expected_series.value)
        np.testing.assert_array_equal(
            actual_series.times.value, expected_series.times.value
        )
        assert actual_series.t0 == expected_series.t0

    assert len(queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in queries[0]
    assert payload_rows == [64]


def test_sdb_window_keeps_float_dtype_inferred_from_outside_source_rows(tmp_path):
    path = tmp_path / "mixed-storage.sdb"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (1_700_000_000 + index * 300, value)
                for index, value in enumerate(
                    [1.5, None, "not-a-number"] + list(range(3, 64))
                )
            ],
        )

    with pytest.warns(UserWarning, match="1 non-numeric value"):
        full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    start = float(full["outTemp"].times[20].value)
    end = float(full["outTemp"].times[24].value)
    expected = apply_time_selection(full, start, end)["outTemp"]

    with pytest.warns(UserWarning, match="1 non-numeric value"):
        result = sdb.read_timeseriesdict_sdb(
            path,
            columns=["outTemp"],
            start=start,
            end=end,
        )["outTemp"]

    assert result.dtype == expected.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result.value, expected.value)
    np.testing.assert_array_equal(result.times.value, expected.times.value)


def test_sdb_payload_warning_precedes_static_timestamp_error(tmp_path):
    path = tmp_path / "warning-before-cadence.sdb"
    _create_wal_database(path, include_units=False)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE archive SET outTemp = 'not-a-number' WHERE rowid = 1"
        )
        connection.execute("UPDATE archive SET dateTime = dateTime + 1 WHERE rowid = 2")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="expected cadence 300, got 301"):
            sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])

    assert [str(item.message) for item in caught] == [
        "SDB column 'outTemp': 1 non-numeric value(s) could not be parsed and "
        "were set to NaN."
    ]


def test_sdb_bounded_invalid_timestamp_discards_scanner_warning_count(
    tmp_path, monkeypatch
):
    path = tmp_path / "bounded-warning-before-invalid-timestamp.sdb"
    _create_wal_database(path, include_units=False)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE archive SET outTemp = 'not-a-number' WHERE rowid = 64"
        )
        connection.execute("UPDATE archive SET dateTime = 'invalid' WHERE rowid = 2")

    original_to_numeric = pd.to_numeric
    conversion_sizes: list[int] = []

    def record_conversion(values, *args, **kwargs):
        conversion_sizes.append(len(values))
        return original_to_numeric(values, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "to_numeric", record_conversion)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError,
            match="SDB dateTime at index 1 must contain integer Unix seconds",
        ):
            sdb.read_timeseriesdict_sdb(
                path,
                columns=["outTemp"],
                start=1_384_035_218.0,
                end=1_384_035_518.0,
            )

    assert [str(item.message) for item in caught] == [
        "SDB column 'outTemp': 1 non-numeric value(s) could not be parsed and "
        "were set to NaN."
    ]
    # This is a unit-conversion column with TEXT storage, so preflight skips
    # pandas coercion and lets the legacy full-payload route own warning order.
    assert conversion_sizes == [64]


def test_sdb_bounded_payload_warnings_follow_requested_column_order(tmp_path):
    path = tmp_path / "bounded-warning-order.sdb"
    _create_wal_database(path, include_units=False)
    full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    start = float(full["outTemp"].times[20].value)
    end = float(full["outTemp"].times[22].value)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE archive SET outTemp = 'bad-temperature' WHERE rowid = 1"
        )
        connection.execute(
            "UPDATE archive SET outHumidity = 'bad-humidity' WHERE rowid = 2"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sdb.read_timeseriesdict_sdb(
            path,
            columns=["outHumidity", "outTemp"],
            start=start,
            end=end,
        )

    assert [str(item.message) for item in caught] == [
        "SDB column 'outHumidity': 1 non-numeric value(s) could not be parsed and "
        "were set to NaN.",
        "SDB column 'outTemp': 1 non-numeric value(s) could not be parsed and "
        "were set to NaN.",
    ]


def test_sdb_bounded_warning_scan_includes_blob_values_outside_window(tmp_path):
    path = tmp_path / "bounded-blob-warning.sdb"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, payload)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (
                    1_700_000_000 + index * 300,
                    sqlite3.Binary(b"not-a-number") if index == 0 else 60.0 + index,
                )
                for index in range(64)
            ],
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sdb.read_timeseriesdict_sdb(
            path,
            columns=["payload"],
            start=1_384_035_218.0 + 300 * 20,
            end=1_384_035_218.0 + 300 * 22,
        )

    assert len(result["payload"]) == 2
    assert [str(item.message) for item in caught] == [
        "SDB column 'payload': 1 non-numeric value(s) could not be parsed and "
        "were set to NaN."
    ]


def test_sdb_duplicate_columns_use_full_payload_fallback(tmp_path, monkeypatch):
    path = tmp_path / "duplicate-selected-columns.sdb"
    _create_wal_database(path, include_units=False)
    original_read_sql_query = pd.read_sql_query
    queries: list[str] = []

    def record_payload(query, connection, *args, **kwargs):
        queries.append(query)
        return original_read_sql_query(query, connection, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload)
    with pytest.raises(
        TypeError,
        match="^arg must be a list, tuple, 1-d array, or Series$",
    ):
        sdb.read_timeseriesdict_sdb(
            path,
            columns=["outTemp", "outTemp"],
            start=1_384_035_218.0 + 300 * 20,
            end=1_384_035_218.0 + 300 * 22,
        )

    assert len(queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in queries[0]


def test_sdb_missing_column_with_invalid_timestamp_uses_legacy_error_order(
    tmp_path, monkeypatch
):
    path = tmp_path / "missing-selected-column.sdb"
    _create_wal_database(path, include_units=False)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE archive SET dateTime = 'invalid' WHERE rowid = 2")

    original_read_sql_query = pd.read_sql_query
    queries: list[str] = []

    def record_payload(query, connection, *args, **kwargs):
        queries.append(query)
        return original_read_sql_query(query, connection, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError,
            match="SDB dateTime at index 1 must contain integer Unix seconds",
        ):
            sdb.read_timeseriesdict_sdb(
                path,
                columns=["missing"],
                start=1_384_035_218.0,
                end=1_384_035_518.0,
            )

    assert [str(item.message) for item in caught] == [
        "SDB column '\"missing\"': 64 non-numeric value(s) could not be parsed and "
        "were set to NaN."
    ]
    assert len(queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in queries[0]


def test_sdb_units_error_precedes_missing_selected_column(tmp_path, monkeypatch):
    path = tmp_path / "units-before-missing-column.sdb"
    _create_wal_database(path, include_units=True)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE archive SET usUnits = 2 WHERE rowid = 1")

    def reject_payload_query(*args, **kwargs):
        pytest.fail("invalid units must fail before payload SQL")

    monkeypatch.setattr(sdb.pd, "read_sql_query", reject_payload_query)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as error:
            sdb.read_timeseriesdict_sdb(
                path,
                columns=["missing"],
                start=1_384_035_218.0,
                end=1_384_035_518.0,
            )

    assert str(error.value) == (
        "SDB usUnits validation failed at dateTime 1700000000: "
        "value 2 must be integer 1."
    )
    assert caught == []


def test_sdb_irregular_cadence_median_avoids_sqlite_integer_overflow(tmp_path):
    path = tmp_path / "int64-delta-overflow.sdb"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp REAL)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [(-(2**63), 60.0), (1, 61.0), (2, 62.0)],
        )

    with pytest.raises(ValueError) as caught:
        TimeSeriesDict.read(path, format="sdb", columns=["outTemp"])

    assert str(caught.value) == (
        "SDB timestamp gap at index 2: expected cadence 9223372036854775809, got 1"
    )


@pytest.mark.parametrize("source_dt", [1 / 3, 0.1])
def test_sdb_window_falls_back_for_noninteger_cadence(source_dt: float) -> None:
    # SDB's integer dateTime validation makes these cadences unsupported. The
    # generic TimeSeries xindex may differ by one ULP from scalar t0+i*dt, so
    # only integer-second SDB grids use the binary-search pushdown path.
    assert (
        sdb._sdb_payload_window(
            1_700_000_000,
            16,
            source_dt,
            1_384_035_218.0,
            1_384_035_219.0,
            1_384_035_220.0,
        )
        is None
    )


def test_sdb_warning_scan_vectorizes_bounded_text_chunks(tmp_path, monkeypatch) -> None:
    path = tmp_path / "chunked-warning-scan.sdb"
    rows = 530
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, payload)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (
                    1_700_000_000 + index,
                    "1.25" if index % 100 == 0 else "not-a-number",
                )
                for index in range(rows)
            ],
        )

    malformed_count = rows - len(range(0, rows, 100))
    with pytest.warns(UserWarning, match=f"{malformed_count} non-numeric value"):
        full = sdb.read_timeseriesdict_sdb(path, columns=["payload"])
    start = float(full["payload"].times[0].value)
    end = float(full["payload"].times[2].value)
    expected = apply_time_selection(full, start, end)["payload"]

    original_to_numeric = pd.to_numeric
    input_sizes: list[int] = []

    def record_vector_input(values, *args, **kwargs):
        assert isinstance(values, pd.Series)
        input_sizes.append(len(values))
        return original_to_numeric(values, *args, **kwargs)

    original_scan = sdb._scan_sdb_timestamps
    scan_warning_columns: list[list[str] | None] = []

    def record_timestamp_scan(
        connection,
        table_identifier,
        order_clause,
        warning_columns=None,
        unit_conversion_columns=None,
    ):
        scan_warning_columns.append(warning_columns)
        return original_scan(
            connection,
            table_identifier,
            order_clause,
            warning_columns=warning_columns,
            unit_conversion_columns=unit_conversion_columns,
        )

    monkeypatch.setattr(sdb.pd, "to_numeric", record_vector_input)
    monkeypatch.setattr(sdb, "_scan_sdb_timestamps", record_timestamp_scan)
    with pytest.warns(UserWarning, match=f"{malformed_count} non-numeric value"):
        result = sdb.read_timeseriesdict_sdb(
            path,
            columns=["payload"],
            start=start,
            end=end,
        )["payload"]

    assert input_sizes == [256, 256, 18, len(expected)]
    assert scan_warning_columns == [["payload"]]
    np.testing.assert_array_equal(result.value, expected.value)
    np.testing.assert_array_equal(result.times.value, expected.times.value)


def test_sdb_numeric_storage_fast_path_matches_full_crop(tmp_path, monkeypatch) -> None:
    path = tmp_path / "numeric-storage-fast-path.sdb"
    rows = 600
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (
                    1_700_000_000 + index * 300,
                    index if index < 16 else None if index == 40 else index + 0.25,
                )
                for index in range(rows)
            ],
        )

    full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    start = float(full["outTemp"].times[10].value)
    end = float(full["outTemp"].times[12].value)
    expected = apply_time_selection(full, start, end)["outTemp"]
    assert expected.dtype == np.dtype(np.float64)

    original_to_numeric = pd.to_numeric
    conversion_sizes: list[int] = []

    def record_vector_input(values, *args, **kwargs):
        conversion_sizes.append(len(values))
        return original_to_numeric(values, *args, **kwargs)

    original_scan = sdb._scan_sdb_timestamps
    scan_warning_columns: list[list[str] | None] = []
    unit_chunk_sizes: list[int] = []
    original_unit_chunk_check = sdb._unit_conversion_chunk_has_fp_events
    original_read_sql_query = pd.read_sql_query
    payload_rows: list[int] = []
    payload_queries: list[str] = []

    def record_payload_query(query, connection, *args, **kwargs):
        payload_queries.append(query)
        frame = original_read_sql_query(query, connection, *args, **kwargs)
        payload_rows.append(len(frame))
        return frame

    def record_unit_chunk_check(column, values):
        unit_chunk_sizes.append(len(values))
        return original_unit_chunk_check(column, values)

    def record_timestamp_scan(
        connection,
        table_identifier,
        order_clause,
        warning_columns=None,
        unit_conversion_columns=None,
    ):
        scan_warning_columns.append(warning_columns)
        return original_scan(
            connection,
            table_identifier,
            order_clause,
            warning_columns=warning_columns,
            unit_conversion_columns=unit_conversion_columns,
        )

    monkeypatch.setattr(sdb.pd, "to_numeric", record_vector_input)
    monkeypatch.setattr(sdb, "_scan_sdb_timestamps", record_timestamp_scan)
    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload_query)
    monkeypatch.setattr(
        sdb, "_unit_conversion_chunk_has_fp_events", record_unit_chunk_check
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sdb.read_timeseriesdict_sdb(
            path,
            columns=["outTemp"],
            start=start,
            end=end,
        )["outTemp"]

    assert not [item for item in caught if issubclass(item.category, UserWarning)]
    assert conversion_sizes == [len(expected)]
    assert unit_chunk_sizes == [256, 256, rows - 512]
    assert max(unit_chunk_sizes) <= sdb._SDB_WARNING_SCAN_CHUNK_ROWS
    assert scan_warning_columns == [["outTemp"]]
    assert len(payload_queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' in payload_queries[0]
    assert payload_rows == [len(expected)]
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected.value)
    np.testing.assert_array_equal(result.times.value, expected.times.value)


def test_sdb_canonical_integer_text_unit_window_materializes_only_requested_rows(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "canonical-integer-text-fast-path.sdb"
    rows = 600
    values: list[object] = [str(index - 300) for index in range(rows)]
    values[3] = 62.5
    values[17] = None
    values[255] = 100.25
    values[256] = None
    values[511] = 70.75
    values[512] = None
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (1_700_000_000 + index * 300, value)
                for index, value in enumerate(values)
            ],
        )

    full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    start = float(full["outTemp"].times[254].value)
    end = float(full["outTemp"].times[260].value)
    expected = apply_time_selection(full, start, end)["outTemp"]

    original_to_numeric = pd.to_numeric
    conversion_sizes: list[int] = []

    def record_vector_input(values, *args, **kwargs):
        conversion_sizes.append(len(values))
        return original_to_numeric(values, *args, **kwargs)

    original_read_sql_query = pd.read_sql_query
    payload_rows: list[int] = []
    payload_queries: list[str] = []

    def record_payload_query(query, connection, *args, **kwargs):
        payload_queries.append(query)
        frame = original_read_sql_query(query, connection, *args, **kwargs)
        payload_rows.append(len(frame))
        return frame

    original_unit_chunk_check = sdb._unit_conversion_chunk_has_fp_events
    unit_chunk_sizes: list[int] = []

    def record_unit_chunk_check(column, values):
        unit_chunk_sizes.append(len(values))
        return original_unit_chunk_check(column, values)

    monkeypatch.setattr(sdb.pd, "to_numeric", record_vector_input)
    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload_query)
    monkeypatch.setattr(
        sdb, "_unit_conversion_chunk_has_fp_events", record_unit_chunk_check
    )

    result = sdb.read_timeseriesdict_sdb(
        path,
        columns=["outTemp"],
        start=start,
        end=end,
    )["outTemp"]

    assert len(payload_queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' in payload_queries[0]
    assert payload_rows == [len(expected)]
    assert unit_chunk_sizes == [256, 256, rows - 512]
    assert max(unit_chunk_sizes) <= sdb._SDB_WARNING_SCAN_CHUNK_ROWS
    # The text preflight parses its strict token grammar directly. Pandas
    # coercion applies only to the requested DataFrame payload.
    assert conversion_sizes == [len(expected)]
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected.value)
    np.testing.assert_array_equal(result.times.value, expected.times.value)


def test_sdb_negative_zero_unit_text_uses_full_payload_fallback(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "negative-zero-unit-text.sdb"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outHumidity)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (1_700_000_000, "-0"),
                (1_700_000_300, 1.5),
                (1_700_000_600, "2"),
            ],
        )

    full = sdb.read_timeseriesdict_sdb(path, columns=["outHumidity"])
    start = float(full["outHumidity"].times[0].value)
    end = float(full["outHumidity"].times[1].value)
    expected = apply_time_selection(full, start, end)["outHumidity"]

    original_read_sql_query = pd.read_sql_query
    payload_queries: list[str] = []

    def record_payload_query(query, connection, *args, **kwargs):
        payload_queries.append(query)
        return original_read_sql_query(query, connection, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload_query)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sdb.read_timeseriesdict_sdb(
            path,
            columns=["outHumidity"],
            start=start,
            end=end,
        )["outHumidity"]

    assert not caught
    assert len(payload_queries) == 1
    assert result.dtype == expected.dtype == np.dtype(np.float64)
    assert result.value.tobytes() == expected.value.tobytes()
    assert np.signbit(result.value).tolist() == [True]
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in payload_queries[0]


@pytest.mark.parametrize(
    ("case", "bad_value", "bad_index", "warn_count"),
    [
        ("decimal", "61.5", 40, 0),
        ("whitespace", " 61 ", 40, 0),
        ("trailing-newline", "61\n", 10, 0),
        ("invalid-outside", "not-a-number", 40, 1),
        ("invalid-selected", "not-a-number", 10, 1),
        ("uint64-edge", "18446744073709551615", 40, 0),
        ("blob-outside", sqlite3.Binary(bytes([255])), 40, 1),
    ],
)
def test_sdb_unqualified_unit_text_uses_full_payload_fallback(
    tmp_path, monkeypatch, case, bad_value, bad_index, warn_count
) -> None:
    path = tmp_path / f"unqualified-unit-text-{case}.sdb"
    rows = 64
    values: list[object] = [str(index + 60) for index in range(rows)]
    values[bad_index] = bad_value
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (1_700_000_000 + index * 300, value)
                for index, value in enumerate(values)
            ],
        )

    original_read_sql_query = pd.read_sql_query
    payload_rows: list[int] = []
    payload_queries: list[str] = []

    def record_payload_query(query, connection, *args, **kwargs):
        payload_queries.append(query)
        frame = original_read_sql_query(query, connection, *args, **kwargs)
        payload_rows.append(len(frame))
        return frame

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload_query)

    start = float(sdb.Time(1_700_000_000 + 10 * 300, format="unix").gps)
    end = float(sdb.Time(1_700_000_000 + 12 * 300, format="unix").gps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sdb.read_timeseriesdict_sdb(
            path,
            columns=["outTemp"],
            start=start,
            end=end,
        )["outTemp"]

    user_warnings = [item for item in caught if issubclass(item.category, UserWarning)]
    assert len(user_warnings) == warn_count
    assert len(payload_queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in payload_queries[0]
    assert payload_rows == [rows]
    assert len(result) == 2
    assert len(result.times) == 2
    if case == "invalid-selected":
        assert np.isnan(result.value[0])


@pytest.mark.parametrize(
    ("column", "bad_value", "operation", "message"),
    [
        ("barometer", 1e308, "over", "overflow encountered in multiply"),
        ("windSpeed", 1e-320, "under", "underflow encountered in multiply"),
    ],
)
@pytest.mark.parametrize("mode", ["warn", "raise"])
def test_sdb_window_replays_out_of_window_unit_conversion_fp_events(
    tmp_path, monkeypatch, column, bad_value, operation, message, mode
) -> None:
    path = tmp_path / f"unit-conversion-{column}-{operation}-{mode}.sdb"
    with sqlite3.connect(path) as connection:
        connection.execute(f'CREATE TABLE archive (dateTime INTEGER, "{column}" REAL)')
        values = [30.0 if column == "barometer" else 10.0] * 64
        values[0] = bad_value
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (1_700_000_000 + index * 300, value)
                for index, value in enumerate(values)
            ],
        )

    original_read_sql_query = pd.read_sql_query
    queries: list[str] = []

    def record_payload_query(query, connection, *args, **kwargs):
        queries.append(query)
        return original_read_sql_query(query, connection, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload_query)
    start = 1_384_035_218.0 + 20 * 300
    end = 1_384_035_218.0 + 24 * 300
    settings = {
        "over": "ignore",
        "under": "ignore",
        "invalid": "ignore",
        "divide": "ignore",
    }
    settings[operation] = mode

    with np.errstate(**settings):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if mode == "raise":
                with pytest.raises(FloatingPointError, match=message):
                    sdb.read_timeseriesdict_sdb(
                        path, columns=[column], start=start, end=end
                    )
            else:
                result = sdb.read_timeseriesdict_sdb(
                    path, columns=[column], start=start, end=end
                )
                assert len(result[column]) == 4

    assert len(queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in queries[0]
    if mode == "warn":
        assert [(type(item.message), str(item.message)) for item in caught] == [
            (RuntimeWarning, message)
        ]
    else:
        assert caught == []


def test_sdb_unit_text_storage_uses_full_payload_fallback(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "unit-text-fallback.sdb"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [
                (1_700_000_000 + index * 300, "15.25" if index == 0 else 60.0 + index)
                for index in range(64)
            ],
        )

    full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    start = float(full["outTemp"].t0.value) + 20 * float(full["outTemp"].dt.value)
    end = float(full["outTemp"].t0.value) + 22 * float(full["outTemp"].dt.value)
    expected = apply_time_selection(full, start, end)["outTemp"]
    original_read_sql_query = pd.read_sql_query
    queries: list[str] = []

    def record_payload_query(query, connection, *args, **kwargs):
        queries.append(query)
        return original_read_sql_query(query, connection, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload_query)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sdb.read_timeseriesdict_sdb(
            path, columns=["outTemp"], start=start, end=end
        )["outTemp"]

    assert caught == []
    assert len(queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in queries[0]
    assert result.t0 == expected.t0
    np.testing.assert_array_equal(result.value, expected.value)


def test_sdb_window_falls_back_when_gwpy_dx_roundtrip_changes_cadence(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "dx-roundtrip-fallback.sdb"
    cadence = 2**53 - 1
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp REAL)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [(index * cadence, float(index)) for index in range(8)],
        )

    full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    series = full["outTemp"]
    source_cadence = float(cadence)
    series_dx = float(series.dt.value)
    assert series_dx == float(1.0 / (1.0 / source_cadence))
    assert series_dx != source_cadence
    start = float(series.t0.value) + 3 * series_dx
    end = float(series.t0.value) + 6 * series_dx
    expected = apply_time_selection(full, start, end)["outTemp"]

    original_read_sql_query = pd.read_sql_query
    queries: list[str] = []

    def record_payload_query(query, connection, *args, **kwargs):
        queries.append(query)
        return original_read_sql_query(query, connection, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload_query)
    result = sdb.read_timeseriesdict_sdb(
        path, columns=["outTemp"], start=start, end=end
    )["outTemp"]

    assert len(queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in queries[0]
    assert result.t0 == expected.t0
    np.testing.assert_array_equal(result.value, expected.value)


def test_sdb_window_falls_back_when_exclusive_upper_exceeds_sqlite_int64(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "sqlite-int64-upper-fallback.sdb"
    cadence = 2**53 - 2
    rows = 1025
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp REAL)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [(index * cadence, float(index)) for index in range(rows)],
        )

    full = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"])
    series = full["outTemp"]
    start = float(series.t0.value) + 1020 * float(series.dt.value)
    expected = apply_time_selection(full, start, None)["outTemp"]

    original_read_sql_query = pd.read_sql_query
    queries: list[str] = []

    def record_payload_query(query, connection, *args, **kwargs):
        queries.append(query)
        return original_read_sql_query(query, connection, *args, **kwargs)

    monkeypatch.setattr(sdb.pd, "read_sql_query", record_payload_query)
    result = sdb.read_timeseriesdict_sdb(path, columns=["outTemp"], start=start)[
        "outTemp"
    ]

    assert len(expected) == 5
    assert len(queries) == 1
    assert 'WHERE "dateTime" >= ? AND "dateTime" < ?' not in queries[0]
    assert result.t0 == expected.t0
    np.testing.assert_array_equal(result.value, expected.value)
