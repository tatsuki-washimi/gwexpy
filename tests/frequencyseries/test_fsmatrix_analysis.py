"""Tests for FrequencySeriesMatrix.__new__ and analysis methods."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries.matrix import FrequencySeriesMatrix


def _make_fsm(n_rows=2, n_cols=2, n_freq=64, df=1.0, f0=0.0):
    data = np.random.default_rng(42).normal(size=(n_rows, n_cols, n_freq))
    return FrequencySeriesMatrix(data, df=df * u.Hz, f0=f0 * u.Hz)


# ---------------------------------------------------------------------------
# FrequencySeriesMatrix.__new__ — constructor branches
# ---------------------------------------------------------------------------

def test_basic_construction():
    fsm = _make_fsm()
    assert fsm.shape == (2, 2, 64)


def test_constructor_with_frequencies():
    freqs = np.linspace(0, 100, 50)
    data = np.ones((1, 1, 50))
    fsm = FrequencySeriesMatrix(data, frequencies=freqs)
    assert fsm.shape == (1, 1, 50)


def test_constructor_with_df_no_f0():
    # df given but no f0 → default x0=0
    data = np.ones((1, 1, 32))
    fsm = FrequencySeriesMatrix(data, df=2.0)
    assert fsm.shape == (1, 1, 32)


def test_constructor_with_channel_names_nxm():
    data = np.ones((2, 3, 16))
    names = np.array([["r0c0", "r0c1", "r0c2"], ["r1c0", "r1c1", "r1c2"]])
    fsm = FrequencySeriesMatrix(data, df=1.0, f0=0.0, channel_names=names)
    assert fsm.shape == (2, 3, 16)


def test_constructor_channel_names_row_vector():
    data = np.ones((2, 2, 16))
    # N=2 channel names → broadcasts as (N,1)
    names = ["ch0", "ch1"]
    fsm = FrequencySeriesMatrix(data, df=1.0, f0=0.0, channel_names=names)
    assert fsm.shape == (2, 2, 16)


def test_constructor_channel_names_flat_nxm():
    data = np.ones((2, 2, 16))
    # N*M=4 channel names → reshape to (N,M)
    names = ["r0c0", "r0c1", "r1c0", "r1c1"]
    fsm = FrequencySeriesMatrix(data, df=1.0, f0=0.0, channel_names=names)
    assert fsm.shape == (2, 2, 16)


def test_constructor_channel_names_other_size():
    # Size doesn't match N or N*M → stored as 1D, may raise broadcast error
    data = np.ones((2, 2, 16))
    names = ["ch0", "ch1", "ch2"]
    try:
        fsm = FrequencySeriesMatrix(data, df=1.0, f0=0.0, channel_names=names)
        assert fsm.shape == (2, 2, 16)
    except ValueError:
        pass  # broadcast failure is acceptable for mismatched sizes


def test_constructor_with_xunit():
    data = np.ones((1, 1, 10))
    fsm = FrequencySeriesMatrix(data, df=1.0, f0=0.0, xunit="kHz")
    assert fsm.shape == (1, 1, 10)


# ---------------------------------------------------------------------------
# FrequencySeriesMatrixAnalysisMixin — ifft
# ---------------------------------------------------------------------------

def test_ifft_returns_timeseries_matrix():
    from gwexpy.timeseries import TimeSeriesMatrix
    fsm = _make_fsm(n_freq=65, df=1.0)
    result = fsm.ifft()
    assert isinstance(result, TimeSeriesMatrix)
    # n_freq=65 → nout=(65-1)*2=128
    assert result.shape[-1] == 128


def test_ifft_with_float_df():
    # df as plain float (not Quantity)
    data = np.ones((1, 1, 33))
    fsm = FrequencySeriesMatrix(data, df=1.0, f0=0.0)
    # Manually set df as float for branch coverage
    fsm._dx = 1.0
    result = fsm.ifft()
    assert result is not None


# ---------------------------------------------------------------------------
# FrequencySeriesMatrixAnalysisMixin — apply_response
# ---------------------------------------------------------------------------

def test_apply_response_with_ndarray():
    fsm = _make_fsm(n_freq=16)
    response = np.ones(16)
    result = fsm.apply_response(response)
    assert result.shape == fsm.shape


def test_apply_response_with_quantity():
    fsm = _make_fsm(n_freq=16)
    response = u.Quantity(np.ones(16), u.dimensionless_unscaled)
    result = fsm.apply_response(response)
    assert result.shape == fsm.shape


def test_apply_response_inplace():
    fsm = _make_fsm(n_freq=16)
    response = np.ones(16)
    result = fsm.apply_response(response, inplace=True)
    assert result is not None
    assert result.shape == fsm.shape


# ---------------------------------------------------------------------------
# FrequencySeriesMatrixAnalysisMixin — smooth
# ---------------------------------------------------------------------------

def test_smooth_amplitude():
    fsm = _make_fsm(n_freq=32)
    result = fsm.smooth(5, method="amplitude")
    assert result.shape == fsm.shape
    assert np.all(result.value >= 0)


def test_smooth_power():
    fsm = _make_fsm(n_freq=32)
    result = fsm.smooth(5, method="power")
    assert result.shape == fsm.shape


def test_smooth_db():
    data = np.abs(np.random.default_rng(0).normal(size=(1, 1, 32))) + 1e-10
    fsm = FrequencySeriesMatrix(data, df=1.0 * u.Hz, f0=0.0 * u.Hz)
    result = fsm.smooth(5, method="db")
    assert result.shape == fsm.shape


def test_smooth_complex():
    data = (np.random.default_rng(0).normal(size=(1, 1, 32))
            + 1j * np.random.default_rng(1).normal(size=(1, 1, 32)))
    data = data.astype(complex)
    fsm = FrequencySeriesMatrix(data, df=1.0 * u.Hz, f0=0.0 * u.Hz)
    result = fsm.smooth(5, method="complex")
    assert result.shape == fsm.shape


def test_smooth_ignore_nan_false():
    fsm = _make_fsm(n_freq=32)
    result = fsm.smooth(5, method="amplitude", ignore_nan=False)
    assert result.shape == fsm.shape


def test_smooth_unknown_method_raises():
    fsm = _make_fsm(n_freq=32)
    with pytest.raises(ValueError, match="Unknown smoothing method"):
        fsm.smooth(5, method="bad_method")
