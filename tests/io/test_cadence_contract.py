"""Fail-closed cadence contracts for SDB and numeric CSV inputs."""

import datetime as dt
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from gwexpy.io.utils import _validate_regular_timestamps
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


def _iso_canonical_instants(iso_timestamps: list[str]) -> list[Decimal]:
    """Derive exact UTC instants from ISO text independently of the reader."""
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    instants = []
    for timestamp in iso_timestamps:
        date_text, time_text = timestamp.removesuffix("Z").split("T")
        hour_text, minute_text, second_text = time_text.split(":")
        whole_second_text, dot, fraction_text = second_text.partition(".")
        whole = dt.datetime.fromisoformat(
            f"{date_text}T{hour_text}:{minute_text}:{whole_second_text}+00:00"
        )
        elapsed = whole - epoch
        instant = Decimal(elapsed.days * 86400 + elapsed.seconds)
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


@pytest.mark.parametrize("timestamp", ["NaN", "Inf", "-Inf"])
def test_numeric_csv_rejects_non_finite_timestamps(tmp_path, timestamp):
    path = tmp_path / "non-finite.csv"
    path.write_text(f"0,1\n{timestamp},2\n")

    with pytest.raises(ValueError, match="CSV timestamp.*non-finite"):
        read_timeseriesdict_csv(path)


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


def test_csv_declared_source_rate_is_checked_before_resampling(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("0,1\n1,2\n3,3\n")

    with pytest.raises(ValueError, match="gap"):
        read_timeseriesdict_csv(
            path, config={"format": {"sample_rate": 1.0}}, resample=2.0
        )


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


def test_time_component_csv_preserves_half_microsecond_cadence(tmp_path):
    iso_timestamps = [
        "2100-08-10T00:00:00.0000000Z",
        "2100-08-10T00:00:00.0000005Z",
        "2100-08-10T00:00:00.0000010Z",
    ]
    canonical = _iso_canonical_instants(iso_timestamps)
    assert canonical[0] > Decimal("4000000000")
    assert [right - left for left, right in zip(canonical, canonical[1:])] == [
        Decimal("0.0000005"),
        Decimal("0.0000005"),
    ]
    path = tmp_path / "half-microsecond-components.csv"
    path.write_text(_component_csv(iso_timestamps))

    series = next(
        iter(
            read_timeseriesdict_csv(
                path, config=_config_with_rate(2_000_000.0)
            ).values()
        )
    )

    assert series.dt.value == pytest.approx(5e-7)


@pytest.mark.parametrize("second", ["NaN", "Inf", "-Inf"])
def test_time_component_csv_rejects_non_finite_second(tmp_path, second):
    path = tmp_path / "non-finite-components.csv"
    path.write_text(f"2026,8,10,0,0,0,1\n2026,8,10,0,0,{second},2\n")

    with pytest.raises(ValueError, match="CSV timestamp.*non-finite"):
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
    "source_rate", [0, -1, float("nan"), float("inf"), -float("inf")]
)
def test_csv_rejects_non_finite_or_non_positive_source_rate(tmp_path, source_rate):
    path = tmp_path / "samples.csv"
    path.write_text("0,1\n1,2\n2,3\n")

    with pytest.raises(ValueError, match="source sample rate"):
        read_timeseriesdict_csv(path, config={"format": {"sample_rate": source_rate}})
