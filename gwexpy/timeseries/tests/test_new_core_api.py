import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries


def test_ifft_roundtrip_gwpy_mode():
    ts = TimeSeries(np.random.randn(32), sample_rate=128.0, unit="V", name="a")
    fs = ts.fft(mode="gwpy")
    ts_back = fs.ifft()
    assert isinstance(ts_back, TimeSeries)
    np.testing.assert_allclose(ts_back.value, ts.value, atol=1e-12)


@pytest.mark.parametrize("pad_mode", ["zero", "reflect"])
def test_ifft_roundtrip_transient(pad_mode):
    data = np.arange(16, dtype=float)
    ts = TimeSeries(data, sample_rate=64.0, unit="")
    fs = ts.fft(mode="transient", pad_left=3, pad_right=2, pad_mode=pad_mode)
    ts_back = fs.ifft(mode="auto")
    assert len(ts_back) == len(ts)
    np.testing.assert_allclose(ts_back.value, ts.value, atol=1e-10)


def test_xcorr_simple_peak():
    ts1 = TimeSeries([0, 1, 0, 0], dt=1.0, name="x")
    ts2 = TimeSeries([0, 0, 1, 0], dt=1.0, name="y")
    xc = ts1.xcorr(ts2, normalize="coeff", demean=False)
    lags = xc.times.value
    lag_at_peak = lags[np.argmax(np.abs(xc.value))]
    assert lag_at_peak == -1  # ts2 is delayed by +1 -> peak at lag -1
    assert np.isclose(np.max(np.abs(xc.value)), 1.0)


def test_append_with_gap_padding_and_overlap_error():
    ts1 = TimeSeries([1, 2, 3], dt=1.0, t0=0.0, name="a")
    ts2 = TimeSeries([4, 5, 6], dt=1.0, t0=5.0, name="b")
    out = ts1.append(ts2, inplace=False, gap="pad", pad=-1)
    np.testing.assert_array_equal(out.value, [1, 2, 3, -1, -1, 4, 5, 6])

    ts_overlap = TimeSeries([7, 8], dt=1.0, t0=2.0, name="c")
    with pytest.raises(ValueError):
        _ = ts1.append(ts_overlap, inplace=False)


def test_asfreq_no_interpolation_fill_value():
    times = np.array([0, 1, 4, 5]) * u.s
    data = np.array([0, 1, 4, 5], dtype=float)
    ts = TimeSeries(data, times=times)

    res = ts.asfreq("1s", fill_value=-9)
    assert res.value[2] == -9
    assert res.value[3] == -9

    with pytest.raises(ValueError):
        _ = ts.asfreq("1s", method="interpolate")


def test_asfreq_downsample_skip_indices():
    ts = TimeSeries(np.arange(6, dtype=float), dt=1 * u.s, t0=0 * u.s)
    res = ts.asfreq("2s")
    np.testing.assert_array_equal(res.value, [0.0, 2.0, 4.0])
