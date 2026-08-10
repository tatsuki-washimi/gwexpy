"""Fail-closed cadence contracts for SDB and numeric CSV inputs."""

import sqlite3
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time

from gwexpy.io.utils import _validate_regular_timestamps
from gwexpy.timeseries.io import csv_enhanced
from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv
from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

_COMPONENT_CONFIG = {
    "format": {"timezone": "UTC"},
    "columns": [
        {
            "name": "year",
            "index": 0,
            "role": "time_component",
            "time_component": "year",
        },
        {
            "name": "month",
            "index": 1,
            "role": "time_component",
            "time_component": "month",
        },
        {"name": "day", "index": 2, "role": "time_component", "time_component": "day"},
        {
            "name": "hour",
            "index": 3,
            "role": "time_component",
            "time_component": "hour",
        },
        {
            "name": "minute",
            "index": 4,
            "role": "time_component",
            "time_component": "minute",
        },
        {
            "name": "second",
            "index": 5,
            "role": "time_component",
            "time_component": "second",
        },
        {"name": "value", "index": 6, "role": "data"},
    ],
}


def _config_with_rate(sample_rate: float) -> dict:
    config = {**_COMPONENT_CONFIG, "format": dict(_COMPONENT_CONFIG["format"])}
    config["format"]["sample_rate"] = sample_rate
    return config


def _component_config_with_skip_rows(skip_rows: int) -> dict:
    config = {**_COMPONENT_CONFIG, "format": dict(_COMPONENT_CONFIG["format"])}
    config["format"]["skip_rows"] = skip_rows
    return config


def _iso_canonical_instants(iso_timestamps: list[str]) -> list[Decimal]:
    """Derive continuous GPS instants from ISO text independently of the reader."""
    instants = []
    for timestamp in iso_timestamps:
        date_text, time_text = timestamp.removesuffix("Z").split("T")
        hour_text, minute_text, second_text = time_text.split(":")
        whole_second_text, dot, fraction_text = second_text.partition(".")
        whole = Time(
            f"{date_text}T{hour_text}:{minute_text}:{whole_second_text}",
            format="isot",
            scale="utc",
        )
        instant = whole.to_value("gps", subfmt="decimal").quantize(
            Decimal("0.000000001")
        )
        if dot:
            instant += Decimal(f"0.{fraction_text}")
        instants.append(instant)
    return instants


def _component_csv(iso_timestamps: list[str]) -> str:
    rows = []
    for index, timestamp in enumerate(iso_timestamps):
        date_text, time_text = timestamp.removesuffix("Z").split("T")
        year, month, day = date_text.split("-")
        hour, minute, second = time_text.split(":")
        rows.append(f"{year},{month},{day},{hour},{minute},{second},{index + 1}")
    return "\n".join(rows) + "\n"


def _write_sdb(path: Path, timestamps: list[object]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE archive (dateTime, outTemp REAL)")
        conn.executemany(
            "INSERT INTO archive VALUES (?, 70)", [(value,) for value in timestamps]
        )


@pytest.mark.parametrize(
    ("times", "match"),
    [
        ([0, 1, 1], "duplicate"),
        ([0, 2, 1], "backward"),
        ([0, 1, 3], "gap"),
    ],
)
def test_regular_timestamp_validator_rejects_invalid_grids(times, match):
    with pytest.raises(ValueError, match=match):
        _validate_regular_timestamps(times, source="test")


def test_regular_timestamp_validator_accepts_single_and_two_points():
    assert _validate_regular_timestamps([Decimal("1")], source="test") == 1.0
    assert (
        _validate_regular_timestamps([Decimal("1"), Decimal("1.25")], source="test")
        == 0.25
    )


def test_regular_timestamp_validator_single_point_honors_declared_cadence():
    assert _validate_regular_timestamps(
        [Decimal("1")], source="test", expected_dt=Decimal("0.25")
    ) == pytest.approx(0.25)


def test_regular_timestamp_validator_uses_token_resolution_for_declared_jitter():
    assert _validate_regular_timestamps(
        [Decimal("0.000"), Decimal("1.001"), Decimal("2.000")],
        source="test",
        expected_dt=Decimal("1"),
    ) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="gap"):
        _validate_regular_timestamps(
            [Decimal("0.000"), Decimal("1.003"), Decimal("2.000")],
            source="test",
            expected_dt=Decimal("1"),
        )


def test_regular_timestamp_validator_rejects_ambiguous_half_cadence_tolerance():
    with pytest.raises(ValueError, match="precision is insufficient"):
        _validate_regular_timestamps(
            [Decimal("0.0"), Decimal("1.0"), Decimal("2.0")],
            source="test",
            expected_dt=Decimal("0.4"),
        )


def test_regular_timestamp_validator_allows_declared_cadence_with_token_precision():
    assert _validate_regular_timestamps(
        [Decimal("0"), Decimal("0.333333333"), Decimal("0.666666667")],
        source="test",
        expected_dt=Decimal("1") / Decimal("3"),
    ) == pytest.approx(1.0 / 3.0)


def test_regular_timestamp_validator_accepts_exact_declared_decimal_grid():
    assert _validate_regular_timestamps(
        [Decimal("0.0"), Decimal("0.1"), Decimal("0.2")],
        source="test",
        expected_dt=Decimal("0.1"),
    ) == pytest.approx(0.1)


def test_regular_timestamp_validator_accepts_serialized_float64_grid():
    """A regular float64 grid remains regular after CSV text serialization."""
    tokens = [str(index * 0.1) for index in range(10)]

    assert _validate_regular_timestamps(tokens, source="CSV") == pytest.approx(0.1)


def test_regular_timestamp_validator_rejects_gap_in_serialized_float64_grid():
    tokens = [str(index * 0.1) for index in range(10)]
    del tokens[5]

    with pytest.raises(ValueError, match="gap"):
        _validate_regular_timestamps(tokens, source="CSV")


def test_csv_accepts_serialized_float64_grid(tmp_path):
    path = tmp_path / "float64-grid.csv"
    path.write_text("\n".join(f"{index * 0.1},{index}" for index in range(10)))

    series = next(iter(read_timeseriesdict_csv(path).values()))

    assert series.dt.value == pytest.approx(0.1)
    assert len(series) == 10


def test_csv_preserves_subsecond_cadence_at_large_absolute_epoch(tmp_path):
    path = tmp_path / "large-epoch.csv"
    path.write_text("1700000000.0,1\n1700000000.1,2\n1700000000.2,3\n")

    series = next(iter(read_timeseriesdict_csv(path).values()))

    assert series.dt.value == pytest.approx(0.1)


def test_csv_accepts_quantized_declared_third_second_cadence(tmp_path):
    path = tmp_path / "thirds.csv"
    path.write_text("0,1\n0.333333333,2\n0.666666667,3\n")

    series = next(
        iter(
            read_timeseriesdict_csv(
                path, config={"format": {"sample_rate": 3.0}}
            ).values()
        )
    )

    assert series.dt.value == pytest.approx(1.0 / 3.0)


def test_numeric_csv_accepts_exact_declared_decimal_grid(tmp_path):
    path = tmp_path / "exact-declared-grid.csv"
    path.write_text("0.0,1\n0.1,2\n0.2,3\n")

    series = next(
        iter(
            read_timeseriesdict_csv(
                path, config={"format": {"sample_rate": 10.0}}
            ).values()
        )
    )

    assert series.dt.value == pytest.approx(0.1)


def test_numeric_csv_accepts_declared_jitter_within_token_resolution(tmp_path):
    path = tmp_path / "jitter-within.csv"
    path.write_text("0.000,1\n1.001,2\n2.000,3\n")

    series = next(
        iter(
            read_timeseriesdict_csv(
                path, config={"format": {"sample_rate": 1.0}}
            ).values()
        )
    )

    assert series.dt.value == pytest.approx(1.0)


def test_numeric_csv_rejects_declared_jitter_outside_token_resolution(tmp_path):
    path = tmp_path / "jitter-outside.csv"
    path.write_text("0.000,1\n1.003,2\n2.000,3\n")

    with pytest.raises(ValueError, match="CSV timestamp gap"):
        read_timeseriesdict_csv(path, config={"format": {"sample_rate": 1.0}})


def test_numeric_csv_single_row_honors_declared_rate_or_one_second_fallback(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("1700000000,1\n")

    declared = next(
        iter(
            read_timeseriesdict_csv(
                path, config={"format": {"sample_rate": 4.0}}
            ).values()
        )
    )
    fallback = next(iter(read_timeseriesdict_csv(path).values()))

    assert declared.dt.value == pytest.approx(0.25)
    assert fallback.dt.value == pytest.approx(1.0)


def test_numeric_csv_preserves_half_microsecond_cadence_at_large_epoch(tmp_path):
    path = tmp_path / "half-microsecond.csv"
    path.write_text(
        "1700000000.0000000,1\n1700000000.0000005,2\n1700000000.0000010,3\n"
    )

    series = next(iter(read_timeseriesdict_csv(path).values()))

    assert series.dt.value == pytest.approx(5e-7)
    expected_relative = np.arange(3) * 5e-7
    represented_relative = series.times.value - series.t0.value
    assert np.max(np.abs(represented_relative - expected_relative)) < 2.5e-7


def test_numeric_csv_rejects_accumulating_arange_precision_loss(tmp_path):
    path = tmp_path / "accumulating-arange-error.csv"
    origin = Decimal("1470000000")
    cadence = Decimal("0.0000005")
    path.write_text(
        "\n".join(f"{origin + cadence * index:f},{index}" for index in range(1000))
        + "\n"
    )

    with pytest.raises(ValueError, match="absolute time axis precision"):
        read_timeseriesdict_csv(path)


def test_numeric_csv_rejects_unrepresentable_submicrosecond_absolute_axis(
    tmp_path,
):
    path = tmp_path / "unrepresentable-axis.csv"
    path.write_text("8000000000.000000,0\n8000000000.000001,1\n8000000000.000002,2\n")

    with pytest.raises(ValueError, match="absolute time axis precision"):
        read_timeseriesdict_csv(
            path,
            config={"format": {"sample_rate": 1_000_000.0}},
            resample=2_000_000.0,
        )


@pytest.mark.parametrize("timestamp", ["NaN", "Inf", "-Inf"])
def test_numeric_csv_rejects_non_finite_timestamps(tmp_path, timestamp):
    path = tmp_path / "non-finite.csv"
    path.write_text(f"0,1\n{timestamp},2\n")

    with pytest.raises(ValueError, match=r"CSV line 2.*timestamp.*non-finite"):
        read_timeseriesdict_csv(path)


@pytest.mark.parametrize("interval", ["1e-400", "1e400"])
def test_numeric_csv_rejects_unrepresentable_float_interval(tmp_path, interval):
    path = tmp_path / "unrepresentable-interval.csv"
    path.write_text(f"0,1\n{interval},2\n")

    with pytest.raises(ValueError, match="absolute time axis"):
        read_timeseriesdict_csv(path)


def test_numeric_csv_rejects_collapsed_relative_axis_before_resampling(tmp_path):
    path = tmp_path / "collapsed-relative-axis.csv"
    path.write_text("0,1\n1e-400,2\n2e-400,3\n")

    with pytest.raises(ValueError, match="absolute time axis"):
        read_timeseriesdict_csv(path, resample=1.0)


def test_numeric_csv_rejects_nat_timestamp_with_line_number(tmp_path):
    path = tmp_path / "nat.csv"
    path.write_text("0,1\nNaT,2\n")

    with pytest.raises(ValueError, match=r"CSV line 2.*non-numeric"):
        read_timeseriesdict_csv(path)


def test_csv_rejects_post_header_non_numeric_row_with_line_number(tmp_path):
    path = tmp_path / "bad-row.csv"
    path.write_text("time,value\n0,1\nnot-a-time,2\n2,3\n")

    with pytest.raises(ValueError, match=r"CSV line 3.*non-numeric"):
        read_timeseriesdict_csv(path)


@pytest.mark.parametrize(("bad_row", "actual_width"), [("1,3", 2), ("1,3,4,5", 4)])
def test_csv_rejects_inconsistent_row_width_with_physical_line_number(
    tmp_path, bad_row, actual_width
):
    path = tmp_path / "ragged.csv"
    path.write_text(f"# generated\ntime,a,b\n0,1,2\n{bad_row}\n")

    with pytest.raises(
        ValueError,
        match=rf"CSV line 4.*{actual_width} columns.*expected 3",
    ):
        read_timeseriesdict_csv(path)


def test_csv_rejects_row_missing_configured_column_with_physical_line_number(
    tmp_path,
):
    path = tmp_path / "configured-short-row.csv"
    path.write_text("# generated\ntime,value,required\n0,1\n")
    config = {
        "format": {"skip_rows": 2},
        "columns": [
            {"name": "time", "index": 0, "role": "time"},
            {"name": "required", "index": 2, "role": "data"},
        ],
    }

    with pytest.raises(
        ValueError,
        match=r"CSV line 3.*2 columns.*configured columns require at least 3",
    ):
        read_timeseriesdict_csv(path, config=config)


@pytest.mark.parametrize("timestamps", [[1, 2, 4], [1, 2, 2]])
def test_sdb_rejects_irregular_timestamp_grid(tmp_path, timestamps):
    db = tmp_path / "irregular.sdb"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE archive (dateTime INTEGER, outTemp REAL)")
        conn.executemany(
            "INSERT INTO archive VALUES (?, 70)", [(t,) for t in timestamps]
        )

    with pytest.raises(ValueError, match="SDB.*(gap|duplicate|backward)"):
        read_timeseriesdict_sdb(db)


def test_sdb_rejects_non_integer_unix_seconds(tmp_path):
    db = tmp_path / "fractional.sdb"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE archive (dateTime, outTemp REAL)")
        conn.executemany("INSERT INTO archive VALUES (?, 70)", [(1.0,), (2.5,)])

    with pytest.raises(ValueError, match="integer Unix seconds"):
        read_timeseriesdict_sdb(db)


def test_sdb_accepts_regular_integer_seconds_at_large_epoch(tmp_path):
    db = tmp_path / "large-epoch.sdb"
    _write_sdb(db, [4_000_000_000, 4_000_000_001, 4_000_000_002])

    series = next(iter(read_timeseriesdict_sdb(db).values()))

    assert series.dt.value == pytest.approx(1.0)


@pytest.mark.parametrize(
    "timestamps",
    [
        [100, 101, 103],
        [100, 101, 101],
        [100, 102, 101],
        [100, 101, 10_000],
    ],
    ids=["missing-sample", "duplicate", "backward", "large-gap"],
)
def test_sdb_rejects_every_irregular_integer_grid(tmp_path, timestamps):
    db = tmp_path / "irregular-matrix.sdb"
    _write_sdb(db, timestamps)

    with pytest.raises(ValueError, match="SDB.*(gap|duplicate|backward)"):
        read_timeseriesdict_sdb(db)


def test_sdb_without_rowid_validates_declared_primary_key_order(tmp_path):
    db = tmp_path / "without-rowid-backward.sdb"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE archive ("
            "seq INTEGER PRIMARY KEY, dateTime INTEGER, outTemp REAL"
            ") WITHOUT ROWID"
        )
        conn.executemany(
            "INSERT INTO archive VALUES (?, ?, 70)",
            [(1, 100), (2, 102), (3, 101)],
        )

    with pytest.raises(ValueError, match="SDB backward timestamp"):
        read_timeseriesdict_sdb(db)


def test_sdb_without_rowid_preserves_descending_primary_key_order(tmp_path):
    db = tmp_path / "without-rowid-descending.sdb"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE archive ("
            "seq INTEGER, dateTime INTEGER, outTemp REAL, "
            "PRIMARY KEY (seq DESC)"
            ") WITHOUT ROWID"
        )
        conn.executemany(
            "INSERT INTO archive VALUES (?, ?, 70)",
            [(1, 100), (2, 101), (3, 102)],
        )

    with pytest.raises(ValueError, match="SDB backward timestamp"):
        read_timeseriesdict_sdb(db)


def test_sdb_without_rowid_supports_quoted_collated_composite_key(tmp_path):
    db = tmp_path / "without-rowid-composite.sdb"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE archive ("
            '"group key" TEXT COLLATE NOCASE, "seq key" INTEGER, '
            "dateTime INTEGER, outTemp REAL, "
            'PRIMARY KEY ("group key" DESC, "seq key" ASC)'
            ") WITHOUT ROWID"
        )
        conn.executemany(
            "INSERT INTO archive VALUES (?, ?, ?, 70)",
            [("a", 2, 102), ("B", 1, 100), ("a", 1, 101)],
        )

    series = next(iter(read_timeseriesdict_sdb(db).values()))

    assert series.dt.value == pytest.approx(1.0)
    assert len(series) == 3


def test_sdb_without_rowid_ignores_shadowing_rowid_column(tmp_path):
    db = tmp_path / "without-rowid-shadow.sdb"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE archive ("
            "seq INTEGER PRIMARY KEY, rowid INTEGER, "
            "dateTime INTEGER, outTemp REAL"
            ") WITHOUT ROWID"
        )
        conn.executemany(
            "INSERT INTO archive VALUES (?, ?, ?, 70)",
            [(1, 1, 100), (2, 3, 102), (3, 2, 101)],
        )

    with pytest.raises(ValueError, match="SDB backward timestamp"):
        read_timeseriesdict_sdb(db)


def test_sdb_single_row_preserves_one_second_fallback(tmp_path):
    db = tmp_path / "single.sdb"
    _write_sdb(db, [4_000_000_000])

    series = next(iter(read_timeseriesdict_sdb(db).values()))

    assert series.dt.value == pytest.approx(1.0)


@pytest.mark.parametrize("timestamp", [0.0000005, float("inf"), None, "NaT"])
def test_sdb_rejects_non_integer_or_non_finite_timestamp(tmp_path, timestamp):
    db = tmp_path / "invalid-time.sdb"
    _write_sdb(db, [0, timestamp])

    with pytest.raises(ValueError, match="integer Unix seconds"):
        read_timeseriesdict_sdb(db)


def test_sdb_integer_seconds_are_validated_exactly_beyond_float_precision(tmp_path):
    db = tmp_path / "exact-integer-grid.sdb"
    _write_sdb(db, [0, 2**53, 2 * 2**53 + 1])

    with pytest.raises(ValueError, match="SDB timestamp gap"):
        read_timeseriesdict_sdb(db)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("0,1\n1,2\n3,3\n", "gap"),
        ("0,1\n1,2\n10000,3\n", "gap"),
        ("0,1\n1,2\n1,3\n", "duplicate"),
        ("0,1\n2,2\n1,3\n", "backward"),
    ],
)
def test_numeric_csv_rejects_irregular_source_grid(tmp_path, body, match):
    path = tmp_path / "samples.csv"
    path.write_text(body)

    with pytest.raises(ValueError, match=match):
        read_timeseriesdict_csv(path)


def test_numeric_csv_rejects_non_numeric_first_timestamp(tmp_path):
    path = tmp_path / "nat-first.csv"
    path.write_text("NaT,1\n0,2\n1,3\n")

    with pytest.raises(ValueError, match=r"CSV line 1.*non-numeric"):
        read_timeseriesdict_csv(path)


@pytest.mark.parametrize(
    "first_row", ["NaT,NaT", "NaT,bad", '"NaT","NaT"', '"NaT","bad"']
)
def test_numeric_csv_rejects_nat_first_timestamp_without_numeric_value(
    tmp_path, first_row
):
    path = tmp_path / "nat-first-text.csv"
    path.write_text(f"{first_row}\n0,1\n1,2\n")

    with pytest.raises(ValueError, match=r"CSV line 1.*non-numeric"):
        read_timeseriesdict_csv(path)


def test_csv_declared_source_rate_is_checked_before_resampling(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("0,1\n1,2\n3,3\n")

    with pytest.raises(ValueError, match="gap"):
        read_timeseriesdict_csv(
            path, config={"format": {"sample_rate": 1.0}}, resample=2.0
        )


@pytest.mark.parametrize(
    "target_rate",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
        5e-324,
        True,
        np.bool_(True),
    ],
)
def test_csv_rejects_non_finite_or_non_positive_target_rate(tmp_path, target_rate):
    path = tmp_path / "samples.csv"
    path.write_text("0,0\n1,10\n2,20\n")

    with pytest.raises(ValueError, match="target sample rate"):
        read_timeseriesdict_csv(path, resample=target_rate)


def test_csv_rejects_target_rate_with_non_finite_grid_count(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("0,0\n1,10\n2,20\n")

    with pytest.raises(ValueError, match="resampled output exceeds"):
        read_timeseriesdict_csv(path, resample=1e308)


def test_csv_rejects_resampling_over_total_value_budget(tmp_path, monkeypatch):
    path = tmp_path / "samples.csv"
    path.write_text("0,0\n1,10\n2,20\n")
    monkeypatch.setattr(csv_enhanced, "_MAX_RESAMPLED_VALUES", 4)

    with pytest.raises(ValueError, match="resampled output exceeds"):
        read_timeseriesdict_csv(path, resample=2.0)


def test_csv_resample_budget_counts_only_requested_channels(tmp_path, monkeypatch):
    path = tmp_path / "selected-channel.csv"
    path.write_text("0,0,10\n1,1,11\n2,2,12\n")
    config = {
        "columns": [
            {"name": "time", "index": 0, "role": "time"},
            {"name": "a", "index": 1, "role": "data"},
            {"name": "b", "index": 2, "role": "data"},
        ]
    }
    monkeypatch.setattr(csv_enhanced, "_MAX_RESAMPLED_VALUES", 4)

    selected = read_timeseriesdict_csv(
        path,
        config=config,
        channels=["a"],
        resample=1.5,
    )

    assert list(selected) == ["a"]
    assert len(selected["a"]) == 4


def test_csv_multi_source_resample_budget_is_global(tmp_path, monkeypatch):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("0,0\n1,1\n2,2\n")
    second.write_text("3,3\n4,4\n5,5\n")
    monkeypatch.setattr(csv_enhanced, "_MAX_RESAMPLED_VALUES", 5)

    with pytest.raises(ValueError, match="resampled output exceeds"):
        read_timeseriesdict_csv([first, second], resample=1.0)


def test_csv_multi_source_resample_budget_allows_exact_total(tmp_path, monkeypatch):
    first = tmp_path / "first-exact.csv"
    second = tmp_path / "second-exact.csv"
    first.write_text("0,0\n1,1\n2,2\n")
    second.write_text("3,3\n4,4\n5,5\n")
    monkeypatch.setattr(csv_enhanced, "_MAX_RESAMPLED_VALUES", 6)

    series = next(iter(read_timeseriesdict_csv([first, second], resample=1.0).values()))

    assert len(series) == 6


@pytest.mark.parametrize("target_rate", [1.75, np.nextafter(2.0, 0.0)])
def test_csv_allows_resampling_up_to_total_value_budget(
    tmp_path, monkeypatch, target_rate
):
    path = tmp_path / "samples.csv"
    path.write_text("0,0\n1,10\n2,20\n")
    monkeypatch.setattr(csv_enhanced, "_MAX_RESAMPLED_VALUES", 4)

    series = next(iter(read_timeseriesdict_csv(path, resample=target_rate).values()))
    expected_times = np.arange(4, dtype=float) / target_rate

    assert len(series) == 4
    np.testing.assert_allclose(series.value, expected_times * 10.0)


def test_csv_validates_resample_method_for_single_row(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("0,1\n")

    with pytest.raises(ValueError, match="Unknown resample method"):
        read_timeseriesdict_csv(path, resample=2.0, resample_method="bad")


@pytest.mark.parametrize("target_rate", [1.4, 1.04])
def test_csv_resample_values_follow_declared_target_grid(tmp_path, target_rate):
    path = tmp_path / "samples.csv"
    path.write_text("0,0\n1,10\n2,20\n")

    series = next(iter(read_timeseriesdict_csv(path, resample=target_rate).values()))
    expected_times = np.arange(len(series), dtype=float) / target_rate

    assert series.dt.value == pytest.approx(1.0 / target_rate)
    np.testing.assert_allclose(series.value, expected_times * 10.0)


def test_time_component_csv_accepts_regular_canonical_instants(tmp_path):
    iso_timestamps = [
        "2026-08-10T00:00:00.000000Z",
        "2026-08-10T00:00:01.000000Z",
        "2026-08-10T00:00:02.000000Z",
    ]
    canonical = _iso_canonical_instants(iso_timestamps)
    assert [right - left for left, right in zip(canonical, canonical[1:])] == [
        Decimal("1"),
        Decimal("1"),
    ]
    path = tmp_path / "components.csv"
    path.write_text(_component_csv(iso_timestamps))

    series = next(
        iter(read_timeseriesdict_csv(path, config=_COMPONENT_CONFIG).values())
    )

    assert series.dt.value == pytest.approx(1.0)


def test_time_component_csv_rejects_utc_leap_second_gap(tmp_path):
    iso_timestamps = [
        "2016-12-31T23:59:59Z",
        "2017-01-01T00:00:00Z",
        "2017-01-01T00:00:01Z",
    ]
    canonical = _iso_canonical_instants(iso_timestamps)
    assert [right - left for left, right in zip(canonical, canonical[1:])] == [
        Decimal("2"),
        Decimal("1"),
    ]
    path = tmp_path / "utc-leap-gap.csv"
    path.write_text(_component_csv(iso_timestamps))

    with pytest.raises(ValueError, match="CSV timestamp gap"):
        read_timeseriesdict_csv(path, config=_COMPONENT_CONFIG)


def test_time_component_csv_accepts_declared_jitter_within_token_resolution(tmp_path):
    iso_timestamps = [
        "2026-08-10T00:00:00.000000000Z",
        "2026-08-10T00:00:01.000000001Z",
        "2026-08-10T00:00:02.000000000Z",
    ]
    canonical = _iso_canonical_instants(iso_timestamps)
    assert canonical[1] - canonical[0] == Decimal("1.000000001")
    path = tmp_path / "jitter-within.csv"
    path.write_text(_component_csv(iso_timestamps))

    series = next(
        iter(read_timeseriesdict_csv(path, config=_config_with_rate(1.0)).values())
    )

    assert series.dt.value == pytest.approx(1.0)


def test_time_component_csv_rejects_declared_jitter_outside_token_resolution(tmp_path):
    iso_timestamps = [
        "2026-08-10T00:00:00.000000000Z",
        "2026-08-10T00:00:01.000000003Z",
        "2026-08-10T00:00:02.000000000Z",
    ]
    canonical = _iso_canonical_instants(iso_timestamps)
    assert canonical[1] - canonical[0] == Decimal("1.000000003")
    path = tmp_path / "jitter-outside.csv"
    path.write_text(_component_csv(iso_timestamps))

    with pytest.raises(ValueError, match="CSV timestamp gap"):
        read_timeseriesdict_csv(path, config=_config_with_rate(1.0))


@pytest.mark.parametrize(
    "iso_timestamps",
    [
        [
            "2026-08-10T00:00:00Z",
            "2026-08-10T00:00:01Z",
            "2026-08-10T00:00:03Z",
        ],
        [
            "2026-08-10T00:00:00Z",
            "2026-08-10T00:00:01Z",
            "2026-08-10T00:00:01Z",
        ],
        [
            "2026-08-10T00:00:00Z",
            "2026-08-10T00:00:02Z",
            "2026-08-10T00:00:01Z",
        ],
        [
            "2026-08-10T00:00:00Z",
            "2026-08-10T00:00:01Z",
            "2026-08-10T00:01:41Z",
        ],
    ],
    ids=["missing-sample", "duplicate", "backward", "large-gap"],
)
def test_time_component_csv_rejects_irregular_canonical_instants(
    tmp_path, iso_timestamps
):
    canonical = _iso_canonical_instants(iso_timestamps)
    assert len(canonical) == 3
    path = tmp_path / "irregular-components.csv"
    path.write_text(_component_csv(iso_timestamps))

    with pytest.raises(ValueError, match="CSV.*timestamp"):
        read_timeseriesdict_csv(path, config=_COMPONENT_CONFIG)


def test_time_component_csv_single_row_honors_declared_rate_or_fallback(tmp_path):
    path = tmp_path / "single-components.csv"
    path.write_text(_component_csv(["2026-08-10T00:00:00Z"]))

    declared = next(
        iter(read_timeseriesdict_csv(path, config=_config_with_rate(4.0)).values())
    )
    fallback = next(
        iter(read_timeseriesdict_csv(path, config=_COMPONENT_CONFIG).values())
    )

    assert declared.dt.value == pytest.approx(0.25)
    assert fallback.dt.value == pytest.approx(1.0)


def test_time_component_csv_accepts_exact_declared_decimal_grid(tmp_path):
    path = tmp_path / "exact-declared-components.csv"
    path.write_text(
        _component_csv(
            [
                "2026-08-10T00:00:00.0Z",
                "2026-08-10T00:00:00.1Z",
                "2026-08-10T00:00:00.2Z",
            ]
        )
    )

    series = next(
        iter(read_timeseriesdict_csv(path, config=_config_with_rate(10.0)).values())
    )

    assert series.dt.value == pytest.approx(0.1)


@pytest.mark.parametrize("row_count", [1, 3])
def test_index_csv_honors_declared_source_rate(row_count, tmp_path):
    path = tmp_path / "index-route.csv"
    path.write_text("\n".join(str(index + 1) for index in range(row_count)) + "\n")
    config = {
        "format": {"sample_rate": 4.0},
        "columns": [{"name": "value", "index": 0, "role": "data"}],
    }

    series = next(iter(read_timeseriesdict_csv(path, config=config).values()))

    assert series.dt.value == pytest.approx(0.25)
    assert len(series) == row_count


def test_time_component_csv_rejects_unrepresentable_half_microsecond_axis(tmp_path):
    iso_timestamps = [
        "2100-08-10T00:00:00.0000000Z",
        "2100-08-10T00:00:00.0000005Z",
        "2100-08-10T00:00:00.0000010Z",
    ]
    canonical = _iso_canonical_instants(iso_timestamps)
    assert canonical[0] > Decimal("3000000000")
    assert [right - left for left, right in zip(canonical, canonical[1:])] == [
        Decimal("0.0000005"),
        Decimal("0.0000005"),
    ]
    assert np.spacing(float(canonical[0])) >= 2.5e-7
    path = tmp_path / "half-microsecond-components.csv"
    path.write_text(_component_csv(iso_timestamps))

    with pytest.raises(ValueError, match="absolute time axis precision"):
        read_timeseriesdict_csv(path, config=_config_with_rate(2_000_000.0))


def test_time_component_csv_preserves_representable_half_microsecond_axis(tmp_path):
    iso_timestamps = [
        "2026-08-10T00:00:00.0000000Z",
        "2026-08-10T00:00:00.0000005Z",
        "2026-08-10T00:00:00.0000010Z",
    ]
    canonical = _iso_canonical_instants(iso_timestamps)
    assert np.spacing(float(canonical[0])) < 2.5e-7
    path = tmp_path / "representable-half-microsecond-components.csv"
    path.write_text(_component_csv(iso_timestamps))

    series = next(
        iter(
            read_timeseriesdict_csv(
                path, config=_config_with_rate(2_000_000.0)
            ).values()
        )
    )

    assert series.dt.value == pytest.approx(5e-7)
    expected_relative = np.arange(3) * 5e-7
    represented_relative = series.times.value - series.t0.value
    assert np.max(np.abs(represented_relative - expected_relative)) < 2.5e-7


@pytest.mark.parametrize("second", ["NaN", "Inf", "-Inf"])
def test_time_component_csv_rejects_non_finite_second(tmp_path, second):
    path = tmp_path / "non-finite-components.csv"
    path.write_text(f"2026,8,10,0,0,0,1\n2026,8,10,0,0,{second},2\n")

    with pytest.raises(ValueError, match=r"CSV line 2.*second.*non-finite"):
        read_timeseriesdict_csv(path, config=_COMPONENT_CONFIG)


def test_time_component_csv_rejects_nat_with_line_number(tmp_path):
    path = tmp_path / "nat-components.csv"
    path.write_text("2026,8,10,0,0,0,1\n2026,8,10,0,0,NaT,2\n")

    with pytest.raises(ValueError, match=r"CSV line 2.*non-numeric"):
        read_timeseriesdict_csv(path, config=_COMPONENT_CONFIG)


def test_time_component_csv_validates_source_before_explicit_resampling(tmp_path):
    path = tmp_path / "resample-components.csv"
    path.write_text(
        _component_csv(
            [
                "2026-08-10T00:00:00Z",
                "2026-08-10T00:00:01Z",
                "2026-08-10T00:00:03Z",
            ]
        )
    )

    with pytest.raises(ValueError, match="CSV timestamp gap"):
        read_timeseriesdict_csv(path, config=_COMPONENT_CONFIG, resample=2.0)


@pytest.mark.parametrize(
    ("component", "column_index"),
    [("year", 0), ("month", 1), ("day", 2), ("hour", 3), ("minute", 4)],
)
def test_time_component_csv_rejects_fractional_integer_component_at_physical_line(
    tmp_path, component, column_index
):
    values = ["2026", "8", "10", "0", "0", "1", "1"]
    values[column_index] += ".5"
    path = tmp_path / f"fractional-{component}.csv"
    path.write_text(
        "# generated\n"
        "title\n"
        "year,month,day,hour,minute,second,value\n" + ",".join(values) + "\n"
    )

    with pytest.raises(
        ValueError,
        match=rf"CSV line 4.*'{component}'.*integer",
    ):
        read_timeseriesdict_csv(path, config=_component_config_with_skip_rows(3))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (["10000", "8", "10", "0", "0", "1", "1"], "year"),
        (["2026", "13", "10", "0", "0", "1", "1"], "month"),
        (["2026", "8", "32", "0", "0", "1", "1"], "day"),
        (["2026", "8", "10", "24", "0", "1", "1"], "hour"),
        (["2026", "8", "10", "0", "60", "1", "1"], "minute"),
        (["2026", "8", "10", "0", "0", "60", "1"], "second"),
        (["2026", "2", "30", "0", "0", "1", "1"], "invalid datetime"),
        (["2026", "8", "10", "0", "NaN", "1", "1"], "minute.*non-finite"),
    ],
    ids=[
        "year-range",
        "month-range",
        "day-range",
        "hour-range",
        "minute-range",
        "second-range",
        "invalid-calendar-date",
        "non-finite",
    ],
)
def test_time_component_csv_validation_errors_report_physical_line(
    tmp_path, values, match
):
    path = tmp_path / "invalid-component.csv"
    path.write_text(
        "# generated\n"
        "title\n"
        "year,month,day,hour,minute,second,value\n" + ",".join(values) + "\n"
    )

    with pytest.raises(ValueError, match=rf"CSV line 4.*{match}"):
        read_timeseriesdict_csv(path, config=_component_config_with_skip_rows(3))


@pytest.mark.parametrize(
    "source_rate",
    [0, -1, float("nan"), float("inf"), -float("inf"), np.bool_(True)],
)
def test_csv_rejects_non_finite_or_non_positive_source_rate(tmp_path, source_rate):
    path = tmp_path / "samples.csv"
    path.write_text("0,1\n1,2\n2,3\n")

    with pytest.raises(ValueError, match="source sample rate"):
        read_timeseriesdict_csv(path, config={"format": {"sample_rate": source_rate}})
