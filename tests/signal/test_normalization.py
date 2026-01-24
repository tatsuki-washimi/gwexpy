import numpy as np
import pytest

from gwexpy.signal.normalization import convert_scipy_to_dtt, get_enbw


def test_get_enbw_standard():
    fs = 16384
    n = 16384
    # Boxcar (Uniform) window
    window = np.ones(n)

    # For Boxcar, ENBW = fs * (1/1) = fs / N if discretized?
    # Actually industry standard: ENBW = fs * sum(w^2) / sum(w)^2
    # sum(w^2) = N, sum(w)^2 = N^2
    # ENBW = fs * N / N^2 = fs / N
    enbw = get_enbw(window, fs, mode="standard")
    assert enbw == pytest.approx(fs / n)

    # Hann window ENBW is theoretical 1.5 * RBW = 1.5 * (fs/N)
    # However, Discrete Hann is slightly different. Let's check against a large N.
    from scipy.signal import windows
    w_hann = windows.hann(n)
    enbw_hann = get_enbw(w_hann, fs, mode="standard")
    assert enbw_hann == pytest.approx(1.5 * (fs / n), rel=1e-3)


def test_get_enbw_dtt():
    fs = 16384
    n = 16384
    window = np.ones(n)

    # DTT definition for Boxcar should match standard
    enbw_dtt = get_enbw(window, fs, mode="dtt")
    assert enbw_dtt == pytest.approx(fs / n)

    # For Hann, DTT windowNorm = (0.5)^2 = 0.25 (as mean(hann) = 0.5)
    # DTT ENBW = (fs/N) / 0.25 = 4 * (fs/N)
    # This is different from industry standard 1.5. This is the core reason for this module!
    from scipy.signal import windows
    w_hann = windows.hann(n) # sum is exactly 0.5 * N
    enbw_hann_dtt = get_enbw(w_hann, fs, mode="dtt")

    # mean(hann) = 0.5 => windowNorm = 0.25
    # windowBW = (fs/N) / 0.25 = 4.0 * (fs/N)
    assert enbw_hann_dtt == pytest.approx(4.0 * (fs / n), rel=1e-3)


def test_convert_scipy_to_dtt():
    from scipy.signal import welch, windows
    fs = 1000
    n = 1000
    t = np.arange(10000) / fs
    data = np.random.normal(size=len(t))

    w_name = 'hann'
    nperseg = 1000
    w = windows.get_window(w_name, nperseg)

    f, pxx = welch(data, fs=fs, window=w, nperseg=nperseg, scaling='density')

    pxx_dtt = convert_scipy_to_dtt(pxx, w)

    # Verify the conversion ratio
    # ratio = (sum_w2 * n) / (sum_w^2)
    sum_w2 = np.sum(w**2)
    sum_w = np.sum(w)
    expected_ratio = (sum_w2 * nperseg) / (sum_w**2)

    # Hann: sum_w approx 0.5*N, sum_w2 approx 0.375*N
    # ratio approx (0.375 * N * N) / (0.25 * N^2) = 0.375 / 0.25 = 1.5
    # Wait, so DTT PSD values are 1.5x LARGER than Scipy for Hann?
    # Let's check the math again in dtt_normalization_analysis.md

    assert (pxx_dtt / pxx)[0] == pytest.approx(expected_ratio)
