
import numpy as np
from astropy import units as u
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList
from gwexpy.frequencyseries import FrequencySeriesMatrix

def test_csd_matrix_dict():
    # Setup data
    t = np.linspace(0, 1, 100) * u.s
    # Correlated signals
    s1 = np.sin(2 * np.pi * 10 * t.value)
    s2 = -1 * s1 # Perfectly anti-correlated

    tsd = TimeSeriesDict()
    tsd['a'] = TimeSeries(s1, times=t, name='a')
    tsd['b'] = TimeSeries(s2, times=t, name='b')

    # 1. Basic CSD
    # Shape: (2, 2)
    csd_mat = tsd.csd_matrix(fftlength=0.5, overlap=0)
    assert isinstance(csd_mat, FrequencySeriesMatrix)
    assert csd_mat.shape == (2, 2, len(csd_mat.frequencies))

    # Check diagonal (PSD)
    # C_aa should be positive real
    c_aa = csd_mat['a', 'a']
    assert np.all(c_aa.value.real >= 0)
    assert np.allclose(c_aa.value.imag, 0, atol=1e-10)

    # Check off-diagonal and hermitian
    # C_ab should be conjugate of C_ba
    c_ab = csd_mat['a', 'b']
    c_ba = csd_mat['b', 'a']
    # The helper fills C_ba = conj(C_ab) if hermitian=True
    np.testing.assert_allclose(c_ab.value, c_ba.value.conj(), atol=1e-10)

def test_coherence_matrix_list():
    t = np.linspace(0, 1, 1000) * u.s
    s1 = np.random.randn(len(t))
    s2 = np.random.randn(len(t)) # Independent

    ts1 = TimeSeries(s1, times=t, name='ts1')
    ts2 = TimeSeries(s2, times=t, name='ts2')

    # Fix: Correct initialization for TimeSeriesList (expects *items)
    tsl = TimeSeriesList(ts1, ts2)

    # 1. Coherence Matrix
    coh_mat = tsl.coherence_matrix(fftlength=0.2, overlap=0.1, diagonal_value=1.0)

    assert coh_mat.shape == (2, 2, len(coh_mat.frequencies))

    # Diagonals should be 1.0 (manually filled)
    c_11 = coh_mat[0, 0]
    np.testing.assert_allclose(c_11.value, 1.0)

    # Off-diagonal should be symmetric
    c_12 = coh_mat[0, 1]
    c_21 = coh_mat[1, 0]
    np.testing.assert_allclose(c_12.value, c_21.value, atol=1e-10)

    # Check if real part is dominant (coherence magnitude squared is real)
    # GWpy coherence returns real [0, 1].
    assert np.all(c_12.value <= 1.0)
    assert np.all(c_12.value >= 0.0)

def test_csd_matrix_cross():
    # Test rows x cols
    t = np.linspace(0, 1, 100) * u.s
    s = np.random.randn(len(t))

    tsd1 = TimeSeriesDict({'rows': TimeSeries(s, times=t)})
    tsd2 = TimeSeriesDict({'cols1': TimeSeries(s, times=t), 'cols2': TimeSeries(s, times=t)})

    mat = tsd1.csd_matrix(tsd2, fftlength=0.5)

    # Shape: (1, 2, freq)
    assert mat.shape[:2] == (1, 2)
    # Fix: Check keys of rows/cols
    assert list(mat.rows.keys()) == ['rows']
    assert list(mat.cols.keys()) == ['cols1', 'cols2']

def test_diagonal_options():
    t = np.linspace(0, 1, 100) * u.s
    s = np.random.randn(len(t))
    # Fix: TimeSeriesList init
    tsl = TimeSeriesList(TimeSeries(s, times=t))

    # CSD without diagonal
    mat = tsl.csd_matrix(include_diagonal=False, fftlength=0.5)
    # Should be 0? Helper logic says set to 0.
    assert np.all(mat[0, 0].value == 0)

    # Coherence with computed diagonal (should be 1)
    mat_coh = tsl.coherence_matrix(include_diagonal=True, diagonal_value=None, fftlength=0.5)
    # Coherence of self is 1.
    np.testing.assert_allclose(mat_coh[0, 0].value, 1.0, atol=1e-5)
