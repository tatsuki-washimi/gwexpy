import numpy as np
import pytest

from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.timeseries import TimeSeries


def test_fft_matches_gwpy_default():
    sample_rate = 128.0
    data = np.linspace(0.0, 1.0, 32, endpoint=False)

    gwpy_ts = GwpyTimeSeries(data, sample_rate=sample_rate, unit="")
    gwex_ts = TimeSeries(data, sample_rate=sample_rate, unit="")

    fs_gwpy = gwpy_ts.fft()
    fs_gwex = gwex_ts.fft()

    np.testing.assert_array_equal(fs_gwex.value, fs_gwpy.value)
    np.testing.assert_array_equal(fs_gwex.frequencies.value, fs_gwpy.frequencies.value)
    assert fs_gwex.unit == fs_gwpy.unit
    assert fs_gwex.df == fs_gwpy.df
    assert fs_gwex.f0 == fs_gwpy.f0
    assert fs_gwex.epoch == fs_gwpy.epoch

    nfft = 64
    fs_gwpy_n = gwpy_ts.fft(nfft=nfft)
    fs_gwex_n = gwex_ts.fft(nfft=nfft)

    np.testing.assert_array_equal(fs_gwex_n.value, fs_gwpy_n.value)
    np.testing.assert_array_equal(fs_gwex_n.frequencies.value, fs_gwpy_n.frequencies.value)
    assert fs_gwex_n.df == fs_gwpy_n.df


def test_fft_linear_nfft_selection():
    data = np.arange(8, dtype=float)
    sample_rate = 8.0
    ts = TimeSeries(data, sample_rate=sample_rate)

    fs, info = ts.fft(convolution="linear", other_length=ts.size, return_info=True)

    assert info["n_required"] == 2 * ts.size - 1
    assert info["nfft"] >= info["n_required"]
    assert len(fs) == info["nfft"] // 2 + 1
    expected_df = ts.sample_rate.value / info["nfft"]
    assert np.isclose(fs.df.value, expected_df)


def test_fft_linear_too_short_raises():
    data = np.ones(4)
    ts = TimeSeries(data, sample_rate=4.0)

    with pytest.raises(ValueError, match="required length"):
        ts.fft(convolution="linear", other_length=ts.size, nfft=ts.size)


def test_fft_reflect_padding_smoke():
    data = np.arange(16, dtype=float)
    ts = TimeSeries(data, sample_rate=64.0)

    fs0 = ts.fft(convolution="circular")
    fs1 = ts.fft(convolution="circular", pad="reflect", nfft=ts.size * 2)

    assert len(fs1) != len(fs0)
