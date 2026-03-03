import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


def _make_pair(sample_rate: float = 128.0, n: int = 4096):
    rng = np.random.default_rng(42)
    a = TimeSeries(rng.standard_normal(n), sample_rate=sample_rate)
    b = TimeSeries(rng.standard_normal(n), sample_rate=sample_rate)
    return a, b


def test_transfer_function_accepts_gwpy_positional_fftlength_overlap():
    gwpy_ts = pytest.importorskip("gwpy.timeseries")
    GwpyTimeSeries = gwpy_ts.TimeSeries

    rng = np.random.default_rng(7)
    data_a = rng.standard_normal(4096)
    data_b = rng.standard_normal(4096)

    gx_a = TimeSeries(data_a, sample_rate=128.0)
    gx_b = TimeSeries(data_b, sample_rate=128.0)
    gw_a = GwpyTimeSeries(data_a, sample_rate=128.0)
    gw_b = GwpyTimeSeries(data_b, sample_rate=128.0)

    got = gx_a.transfer_function(gx_b, 1.0, 0.5)
    exp = gw_a.transfer_function(gw_b, 1.0, 0.5)

    np.testing.assert_allclose(got.frequencies.value, exp.frequencies.value, rtol=1e-12)
    np.testing.assert_allclose(got.value, exp.value, rtol=1e-10)


def test_transfer_function_mode_keyword_still_works():
    a, b = _make_pair()
    got = a.transfer_function(b, mode="transient")
    assert got.size > 0


def test_transfer_function_legacy_positional_mode_is_deprecated():
    a, b = _make_pair()
    with pytest.warns(DeprecationWarning, match="positional argument"):
        got = a.transfer_function(b, "transient")
    ref = a.transfer_function(b, mode="transient")
    np.testing.assert_allclose(got.value, ref.value, rtol=1e-12, atol=0.0)


def test_transfer_function_positional_mode_and_keyword_mode_conflict():
    a, b = _make_pair()
    with pytest.raises(TypeError, match="both positionally and via keyword"):
        a.transfer_function(b, "steady", mode="transient")


def test_transfer_function_positional_fftlength_with_keyword_window():
    a, b = _make_pair()
    got = a.transfer_function(b, 1.0, window="hann")
    ref = a.transfer_function(b, fftlength=1.0, window="hann")
    np.testing.assert_allclose(got.value, ref.value, rtol=1e-12, atol=0.0)
