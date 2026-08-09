"""Fail-closed cadence contracts for SDB and numeric CSV inputs."""

import sqlite3
from decimal import Decimal

import pytest

from gwexpy.io.utils import _validate_regular_timestamps
from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv
from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb


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


def test_regular_timestamp_validator_allows_declared_cadence_with_token_precision():
    assert _validate_regular_timestamps(
        [Decimal("0"), Decimal("0.333333333"), Decimal("0.666666667")],
        source="test",
        expected_dt=Decimal("1") / Decimal("3"),
    ) == pytest.approx(1.0 / 3.0)


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


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("0,1\n1,2\n3,3\n", "gap"),
        ("0,1\n1,2\n1,3\n", "duplicate"),
        ("0,1\n2,2\n1,3\n", "backward"),
    ],
)
def test_numeric_csv_rejects_irregular_source_grid(tmp_path, body, match):
    path = tmp_path / "samples.csv"
    path.write_text(body)

    with pytest.raises(ValueError, match=match):
        read_timeseriesdict_csv(path)


def test_csv_declared_source_rate_is_checked_before_resampling(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("0,1\n1,2\n3,3\n")

    with pytest.raises(ValueError, match="gap"):
        read_timeseriesdict_csv(
            path, config={"format": {"sample_rate": 1.0}}, resample=2.0
        )


@pytest.mark.parametrize(
    "source_rate", [0, -1, float("nan"), float("inf"), -float("inf")]
)
def test_csv_rejects_non_finite_or_non_positive_source_rate(tmp_path, source_rate):
    path = tmp_path / "samples.csv"
    path.write_text("0,1\n1,2\n2,3\n")

    with pytest.raises(ValueError, match="source sample rate"):
        read_timeseriesdict_csv(path, config={"format": {"sample_rate": source_rate}})
