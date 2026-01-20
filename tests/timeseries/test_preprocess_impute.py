import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.preprocess import (
    _bfill_numpy,
    _ffill_numpy,
    _impute_1d,
    _limit_mask,
    impute_timeseries,
)


def _expected_limit_mask(nans, limit, direction):
    if limit is None:
        return np.zeros_like(nans, dtype=bool)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return nans.copy()
    mask = np.zeros_like(nans, dtype=bool)
    n = len(nans)
    i = 0
    while i < n:
        if not nans[i]:
            i += 1
            continue
        run_start = i
        while i < n and nans[i]:
            i += 1
        run_end = i
        run_len = run_end - run_start
        if run_len > limit:
            if direction == "forward":
                mask[run_start + limit:run_end] = True
            elif direction == "backward":
                mask[run_start:run_end - limit] = True
            else:
                raise ValueError("direction must be forward or backward")
    return mask


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_limit_mask_exhaustive(direction):
    for n in range(1, 9):
        for bits in range(2**n):
            nans = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool)
            for limit in range(0, n + 1):
                expected = _expected_limit_mask(nans, limit, direction)
                result = _limit_mask(nans, limit, direction=direction)
                np.testing.assert_array_equal(result, expected)


def test_ffill_bfill_numpy_expected_without_pandas():
    try:
        import pandas  # noqa: F401
    except ImportError:
        arr = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])

        ffill_expected = np.array([np.nan, 1.0, 1.0, 1.0, 4.0, 4.0])
        bfill_expected = np.array([1.0, 1.0, 4.0, 4.0, 4.0, np.nan])

        np.testing.assert_allclose(_ffill_numpy(arr, limit=None), ffill_expected, equal_nan=True)
        np.testing.assert_allclose(_bfill_numpy(arr, limit=None), bfill_expected, equal_nan=True)
    else:
        pytest.skip("pandas available; covered by pandas parity tests")


@pytest.mark.parametrize("limit", [None, 1, 2])
def test_ffill_numpy_matches_pandas(limit):
    pd = pytest.importorskip("pandas")
    arr = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    expected = pd.Series(arr).ffill(limit=limit).values
    result = _ffill_numpy(arr, limit=limit)
    np.testing.assert_allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize("limit", [None, 1, 2])
def test_bfill_numpy_matches_pandas(limit):
    pd = pytest.importorskip("pandas")
    arr = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    expected = pd.Series(arr).bfill(limit=limit).values
    result = _bfill_numpy(arr, limit=limit)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_impute_1d_matches_interp1d_real():
    pytest.importorskip("scipy")
    from scipy.interpolate import interp1d

    x = np.arange(5, dtype=float)
    y = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    result = _impute_1d(y.copy(), x, "linear", False, None, limit=None)

    valid = ~np.isnan(y)
    f = interp1d(x[valid], y[valid], kind="linear", bounds_error=False, fill_value="extrapolate")
    expected = y.copy()
    expected[~valid] = f(x[~valid])

    np.testing.assert_allclose(result, expected)


def test_impute_1d_matches_interp1d_complex():
    pytest.importorskip("scipy")
    from scipy.interpolate import interp1d

    x = np.arange(5, dtype=float)
    y = np.array([1 + 1j, np.nan + 1j * np.nan, 3 + 3j, np.nan + 1j * np.nan, 5 + 5j])
    result = _impute_1d(y.copy(), x, "linear", False, None, limit=None)

    valid = ~np.isnan(y)
    f_real = interp1d(x[valid], y[valid].real, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_imag = interp1d(x[valid], y[valid].imag, kind="linear", bounds_error=False, fill_value="extrapolate")
    expected = y.copy()
    expected[~valid] = f_real(x[~valid]) + 1j * f_imag(x[~valid])

    np.testing.assert_allclose(result, expected)


def test_impute_max_gap_reverts_interiors_and_edges():
    data = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])

    result = impute_timeseries(data, method="linear", max_gap=2)

    assert np.isnan(result[0])
    assert np.isnan(result[2])
    assert np.isnan(result[3])
    assert np.isnan(result[5])
    assert result[1] == 1.0
    assert result[4] == 4.0


def test_impute_fast_path_matches_slice_path():
    base = np.arange(5, dtype=float)
    data_common = np.vstack([base.copy(), base.copy()])
    data_common[:, 2] = np.nan

    data_mixed = np.vstack([base.copy(), base.copy()])
    data_mixed[0, 2] = np.nan
    data_mixed[1, 3] = np.nan

    res_common = impute_timeseries(data_common, method="linear", axis=-1)
    res_mixed = impute_timeseries(data_mixed, method="linear", axis=-1)

    expected = np.vstack([base, base])
    np.testing.assert_allclose(res_common, expected)
    np.testing.assert_allclose(res_mixed, expected)
    np.testing.assert_allclose(res_common, res_mixed)


def test_impute_timeseries_preserves_metadata():
    times = np.arange(4) * u.s
    data = np.array([1.0, np.nan, 3.0, np.nan])
    ts = TimeSeries(data, times=times, name="chan", unit=u.m, channel="X")

    result = impute_timeseries(ts, method="linear")

    assert isinstance(result, TimeSeries)
    np.testing.assert_allclose(result.times.value, ts.times.value)
    assert result.t0.unit == ts.t0.unit
    assert result.t0.value == pytest.approx(ts.t0.value)
    assert result.dt.unit == ts.dt.unit
    assert result.dt.value == pytest.approx(ts.dt.value)
    assert result.name == ts.name
    assert result.unit == ts.unit
    assert result.channel == ts.channel



def test_impute_timeseries_preserves_metadata_rebuild():
    data = np.array([1.0, np.nan, 3.0])
    ts = TimeSeries(data, t0=0.0, dt=1.0, name="dummy", unit="m")

    result = impute_timeseries(ts, method="linear")

    assert isinstance(result, TimeSeries)
    assert result.t0.value == ts.t0.value
    assert result.dt.value == ts.dt.value
    assert result.name == ts.name
    assert result.unit == ts.unit
