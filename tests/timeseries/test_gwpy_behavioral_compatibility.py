"""Default numerical compatibility with the corresponding GWpy APIs."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.timeseries import TimeSeries

_EXACT_EPOCH_NS = 1_234_567_890_123_456_789


def _write_exact_hdf5(tmp_path, sample_rate: int):
    path = tmp_path / f"exact-{sample_rate}.hdf5"
    source = TimeSeries(
        np.arange(32, dtype=np.float64),
        t0_ns=_EXACT_EPOCH_NS,
        sample_rate=sample_rate,
        unit="V",
        name="X1:COMPAT",
        channel="X1:COMPAT",
    )
    source.write(path, format="hdf5", path="series")
    return path


def _assert_default_result_equal(
    expected: GwpyTimeSeries, observed: TimeSeries
) -> None:
    np.testing.assert_array_equal(observed.value, expected.value)
    np.testing.assert_array_equal(observed.times.value, expected.times.value)
    assert observed.dtype == expected.dtype
    assert observed.unit == expected.unit
    assert observed.name == expected.name
    assert observed.channel == expected.channel
    assert observed.t0 == expected.t0
    assert observed.x0 == expected.x0
    assert observed.dt == expected.dt
    assert observed.dx == expected.dx
    assert observed.span == expected.span


@pytest.mark.parametrize("sample_rate", [4, 4096, 1_000_000, 10_000_000])
def test_exact_hdf5_default_read_matches_gwpy(tmp_path, sample_rate: int) -> None:
    path = _write_exact_hdf5(tmp_path, sample_rate)

    expected = GwpyTimeSeries.read(path, format="hdf5", path="series")
    observed = TimeSeries.read(path, format="hdf5", path="series")

    _assert_default_result_equal(expected, observed)
    assert observed.t0_gps_ns == _EXACT_EPOCH_NS


@pytest.mark.parametrize("sample_rate", [4, 4096, 1_000_000, 10_000_000])
@pytest.mark.parametrize("boundary_offset", [-1, 0, 1])
@pytest.mark.parametrize("bound", ["start", "end"])
def test_exact_hdf5_default_crop_matches_gwpy(
    tmp_path, sample_rate: int, boundary_offset: int, bound: str
) -> None:
    path = _write_exact_hdf5(tmp_path, sample_rate)
    expected_source = GwpyTimeSeries.read(path, format="hdf5", path="series")
    observed_source = TimeSeries.read(path, format="hdf5", path="series")
    boundary = float(expected_source.t0.value) + 8.0 / sample_rate
    epsilon = max(float(np.spacing(boundary)), 1e-12)
    selected_bound = boundary + boundary_offset * epsilon

    crop_bounds = (selected_bound, None) if bound == "start" else (None, selected_bound)
    expected = expected_source.crop(*crop_bounds)
    observed = observed_source.crop(*crop_bounds)

    _assert_default_result_equal(expected, observed)


@pytest.mark.parametrize("sample_rate", [4, 4096, 1_000_000, 10_000_000])
@pytest.mark.parametrize("boundary_offset", [-1, 0, 1])
@pytest.mark.parametrize("bound", ["start", "end"])
def test_exact_hdf5_default_read_bounds_match_gwpy(
    tmp_path, sample_rate: int, boundary_offset: int, bound: str
) -> None:
    path = _write_exact_hdf5(tmp_path, sample_rate)
    full = GwpyTimeSeries.read(path, format="hdf5", path="series")
    boundary = float(full.t0.value) + 8.0 / sample_rate
    epsilon = max(float(np.spacing(boundary)), 1e-12)
    selected_bound = boundary + boundary_offset * epsilon
    bounds = {bound: selected_bound}

    expected = GwpyTimeSeries.read(
        path,
        format="hdf5",
        path="series",
        **bounds,
    )
    observed = TimeSeries.read(
        path,
        format="hdf5",
        path="series",
        **bounds,
    )

    _assert_default_result_equal(expected, observed)


@pytest.mark.parametrize("sample_rate", [4, 4096, 1_000_000, 10_000_000])
@pytest.mark.parametrize("selection", [slice(8, None), slice(None, 8), slice(4, 12)])
def test_exact_hdf5_default_slice_matches_gwpy(
    tmp_path, sample_rate: int, selection: slice
) -> None:
    path = _write_exact_hdf5(tmp_path, sample_rate)
    expected = GwpyTimeSeries.read(path, format="hdf5", path="series")[selection]
    observed = TimeSeries.read(path, format="hdf5", path="series")[selection]

    _assert_default_result_equal(expected, observed)
