"""Per-segment statistic extraction contracts (#406)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwpy.segments import Segment

from gwexpy.table import SegmentTable
from gwexpy.table.segment import SegmentCell
from gwexpy.timeseries import TimeSeries


def _series_table() -> SegmentTable:
    table = SegmentTable.from_segments([Segment(0, 4), Segment(4, 8)])
    table.add_series_column(
        "signal",
        data=[
            TimeSeries([1.0, 2.0, 3.0, 4.0], sample_rate=1, unit="V"),
            TimeSeries([2.0, 4.0, 6.0, 8.0], sample_rate=1, unit="V"),
        ],
    )
    return table


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [
        ("median", [2.5, 5.0]),
        ("mean", [2.5, 5.0]),
        ("rms", [np.sqrt(7.5), np.sqrt(30.0)]),
        ("max", [4.0, 8.0]),
        ("min", [1.0, 2.0]),
    ],
)
def test_builtin_statistics_preserve_units_and_source(
    statistic: str, expected: list[float]
) -> None:
    table = _series_table()
    result = table.extract_stat(statistic, "signal")
    output = f"signal_{statistic}"

    assert result is not table
    assert output not in table.columns
    assert result._units[output] == u.V
    np.testing.assert_allclose([result.row(i)[output] for i in range(2)], expected)
    assert result._meta[f"{output}_status"].tolist() == ["valid", "valid"]
    np.testing.assert_array_equal(table.row(0)["signal"].value, [1, 2, 3, 4])


def test_custom_callable_receives_payload_and_can_choose_output_name() -> None:
    table = _series_table()
    result = table.extract_stat(
        lambda series: float(np.ptp(series.value)), "signal", out_column="range"
    )
    assert result._meta["range"].tolist() == [3.0, 6.0]
    assert "range" not in table.columns


@pytest.mark.parametrize("parallel", [False, True, 1, 2])
def test_parallel_policy_keeps_row_order(parallel: bool | int) -> None:
    table = _series_table()
    result = table.extract_stat("max", "signal", parallel=parallel)
    assert [result.row(i)["signal_max"] for i in range(2)] == [4.0, 8.0]


def test_missing_status_does_not_become_valid_zero() -> None:
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)])
    table.add_series_column(
        "signal",
        data=[
            SegmentCell.missing(),
            TimeSeries([2.0, 4.0], sample_rate=2, unit="V"),
        ],
    )
    result = table.extract_stat("mean", "signal")
    assert np.isnan(result.row(0)["signal_mean"])
    assert result._meta["signal_mean_status"].tolist() == ["missing", "valid"]
    assert result.row(1)["signal_mean"] == 3.0


def test_extracted_stat_roundtrips_with_unit_metadata(tmp_path) -> None:
    result = _series_table().extract_stat("mean", "signal")
    path = tmp_path / "stat.h5"
    result.write(path)
    loaded = SegmentTable.read(path)
    assert loaded._units["signal_mean"] == u.V
    assert loaded.row(0)["signal_mean"] == 2.5


def test_invalid_statistics_and_nonfinite_data_fail_closed() -> None:
    table = _series_table()
    with pytest.raises(ValueError, match="Unknown statistic"):
        table.extract_stat("quantile", "signal")
    with pytest.raises(ValueError, match="parallel"):
        table.extract_stat("mean", "signal", parallel=0)
    with pytest.raises(ValueError, match="already exists"):
        table.extract_stat("mean", "signal", out_column="span")

    bad = SegmentTable.from_segments([Segment(0, 1)])
    bad.add_series_column("signal", data=[TimeSeries([1.0, np.nan], sample_rate=2)])
    with pytest.raises(ValueError, match="finite"):
        bad.extract_stat("mean", "signal")


def test_custom_callable_must_return_scalar() -> None:
    table = _series_table()
    with pytest.raises(ValueError, match="one scalar"):
        table.extract_stat(lambda series: series.value, "signal")
    with pytest.raises(ValueError, match="finite numeric scalar"):
        table.extract_stat(lambda series: np.nan, "signal")


def test_rms_large_integer_samples_avoids_square_overflow() -> None:
    table = SegmentTable.from_segments([Segment(0, 2)])
    table.add_series_column(
        "signal", data=[np.array([2**62, -(2**62)], dtype=np.int64)], kind="array"
    )
    result = table.extract_stat("rms", "signal")
    assert result.row(0)["signal_rms"] == pytest.approx(float(2**62))


def test_mean_large_finite_samples_avoids_sum_overflow() -> None:
    table = SegmentTable.from_segments([Segment(0, 2)])
    table.add_series_column("signal", data=[np.array([1e308, 1e308])], kind="array")
    result = table.extract_stat("mean", "signal")
    assert result.row(0)["signal_mean"] == pytest.approx(1e308)


def test_compatible_row_units_are_converted_to_one_output_unit() -> None:
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)])
    table.add_series_column(
        "signal",
        data=[
            TimeSeries([1.0, 3.0], sample_rate=2, unit="V"),
            TimeSeries([1000.0, 3000.0], sample_rate=2, unit="mV"),
        ],
    )
    result = table.extract_stat("mean", "signal")
    assert result._units["signal_mean"] == u.V
    assert result._meta["signal_mean"].tolist() == pytest.approx([2.0, 2.0])
