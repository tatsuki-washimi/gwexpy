"""Gap and padding behavior on direct collection readers."""

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


@pytest.mark.parametrize("format", ["zarr", "h5"])
def test_direct_reader_respects_padding(tmp_path, format):
    if format == "zarr":
        pytest.importorskip("zarr")
    source = TimeSeriesDict({"A": TimeSeries([1.0, 2.0], t0=10, dt=1, unit="m")})
    path = tmp_path / ("series.zarr" if format == "zarr" else "series.h5")
    source.write(path, format="zarr" if format == "zarr" else "hdf5")
    result = TimeSeriesDict.read(path, format=format, start=9, end=13, pad=0.0)
    np.testing.assert_array_equal(result["A"].value, [0, 1, 2, 0])
    assert result["A"].t0.value == 9
    assert result["A"].dt.value == 1
    assert result["A"].unit == source["A"].unit
    with pytest.raises(ValueError, match="does not cover"):
        TimeSeriesDict.read(path, format=format, start=9, end=13, gap="raise")
    with pytest.raises(ValueError, match="gap must be"):
        TimeSeriesDict.read(path, format=format, gap="bad")


def test_zarr_multisource_gap_and_outer_padding(tmp_path):
    pytest.importorskip("zarr")
    paths = [tmp_path / "first.zarr", tmp_path / "second.zarr"]
    for path, start, values in zip(paths, [10, 13], [[1.0, 2.0], [4.0, 5.0]]):
        TimeSeriesDict({"A": TimeSeries(values, t0=start, dt=1, unit="m")}).write(
            path, format="zarr"
        )
    result = TimeSeriesDict.read(
        paths, format="zarr", start=9, end=16, gap="pad", pad=0
    )
    np.testing.assert_array_equal(result["A"].value, [0, 1, 2, 0, 4, 5, 0])
    assert result["A"].t0.value == 9
    assert result["A"].dt.value == 1
    assert str(result["A"].unit) == "m"


def test_hdf5_padding_preserves_uncertainty_alignment(tmp_path):
    series = TimeSeries([1.0, 2.0], t0=10, dt=1, unit="m")
    series._set_uncertainty(lower=series.copy(), uncertainty_kind="interval")
    path = tmp_path / "uncertainty.h5"
    TimeSeriesDict({"A": series}).write(path, format="hdf5", layout="group")
    result = TimeSeriesDict.read(path, format="h5", start=9, end=13, pad=0.0)
    assert result["A"].lower is not None
    np.testing.assert_array_equal(result["A"].lower.value, [0, 1, 2, 0])
    np.testing.assert_array_equal(
        result["A"].lower.times.value, result["A"].times.value
    )


def test_hdf5_padding_preserves_quantiles_and_aliases(tmp_path):
    series = TimeSeries([1.0, 2.0], t0=10, dt=1, unit="m")
    series._set_uncertainty(
        quantiles={0.1: series.copy(), 0.9: series.copy()},
        interval_levels=(0.1, 0.9),
        uncertainty_kind="predictive",
    )
    path = tmp_path / "quantiles.h5"
    TimeSeriesDict({"A": series}).write(path, format="hdf5", layout="group")
    result = TimeSeriesDict.read(path, format="h5", start=9, end=13, pad=0.0)
    np.testing.assert_array_equal(result["A"].quantiles[0.1].value, [0, 1, 2, 0])
    assert result["A"].lower is result["A"].quantiles[0.1]
    assert result["A"].interval_levels == (0.1, 0.9)


def test_hdf5_fully_disjoint_padding_does_not_invent_uncertainty(tmp_path):
    series = TimeSeries([1.0, 2.0], t0=10, dt=1, unit="m")
    series._set_uncertainty(lower=series.copy(), uncertainty_kind="interval")
    path = tmp_path / "disjoint.h5"
    TimeSeriesDict({"A": series}).write(path, format="hdf5", layout="group")
    result = TimeSeriesDict.read(path, format="h5", start=14, end=16, pad=0.0)
    assert not result["A"]._has_uncertainty()
