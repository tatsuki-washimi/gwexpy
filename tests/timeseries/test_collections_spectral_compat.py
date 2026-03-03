import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


def _make_series(sample_rate: float = 256.0, n: int = 4096):
    rng = np.random.default_rng(123)
    ts1 = TimeSeries(rng.standard_normal(n), sample_rate=sample_rate)
    ts2 = TimeSeries(rng.standard_normal(n), sample_rate=sample_rate)
    ts3 = TimeSeries(rng.standard_normal(n), sample_rate=sample_rate)
    return ts1, ts2, ts3


def test_timeseriesdict_csd_accepts_positional_fft_args():
    ts1, ts2, other = _make_series()
    tsd = TimeSeriesDict({"a": ts1, "b": ts2})

    got = tsd.csd(other, 1.0, 0.5)
    exp = tsd.csd(other, fftlength=1.0, overlap=0.5)

    for key in exp:
        np.testing.assert_allclose(got[key].value, exp[key].value, rtol=1e-12, atol=0.0)


def test_timeserieslist_coherence_accepts_positional_fft_args():
    ts1, ts2, other = _make_series()
    tsl = TimeSeriesList(ts1, ts2)

    got = tsl.coherence(other, 1.0, 0.5)
    exp = tsl.coherence(other, fftlength=1.0, overlap=0.5)

    for i in range(len(exp)):
        np.testing.assert_allclose(
            got[i].value, exp[i].value, rtol=1e-12, atol=0.0, equal_nan=True
        )


def test_collections_csd_positional_keyword_mix_raises_typeerror():
    ts1, ts2, other = _make_series()
    tsd = TimeSeriesDict({"a": ts1, "b": ts2})
    with pytest.raises(TypeError, match="cannot mix positional"):
        tsd.csd(other, 1.0, overlap=0.5)

    tsl = TimeSeriesList(ts1, ts2)
    with pytest.raises(TypeError, match="at most two positional"):
        tsl.csd(other, 1.0, 0.5, 0.25)


def test_collections_numeric_other_is_not_misparsed_as_fftlength():
    ts1, ts2, _ = _make_series()
    tsd = TimeSeriesDict({"a": ts1, "b": ts2})
    with pytest.raises(TypeError, match="other must be"):
        tsd.csd(1.0, 0.5)

    tsl = TimeSeriesList(ts1, ts2)
    with pytest.raises(TypeError, match="other must be"):
        tsl.coherence(1.0, 0.5)
