import numpy as np
import pytest

from gwexpy.signal.normalization import (
    convert_scipy_to_dtt,
    get_enbw,
    get_psd_normalization_factor,
    get_window_normalized,
)


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

    w_hann = windows.hann(n)  # sum is exactly 0.5 * N
    enbw_hann_dtt = get_enbw(w_hann, fs, mode="dtt")

    # mean(hann) = 0.5 => windowNorm = 0.25
    # windowBW = (fs/N) / 0.25 = 4.0 * (fs/N)
    assert enbw_hann_dtt == pytest.approx(4.0 * (fs / n), rel=1e-3)


def test_convert_scipy_to_dtt():
    from scipy.signal import welch, windows

    fs = 1000
    t = np.arange(10000) / fs
    data = np.random.normal(size=len(t))

    w_name = "hann"
    nperseg = 1000
    w = windows.get_window(w_name, nperseg)

    f, pxx = welch(data, fs=fs, window=w, nperseg=nperseg, scaling="density")

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


# ---------------------------------------------------------------------------
# get_enbw — zero window
# ---------------------------------------------------------------------------

def test_get_enbw_zero_window_standard():
    window = np.zeros(100)
    assert get_enbw(window, fs=100.0, mode="standard") == 0.0


def test_get_enbw_zero_window_dtt():
    window = np.zeros(100)
    assert get_enbw(window, fs=100.0, mode="dtt") == 0.0


# ---------------------------------------------------------------------------
# get_psd_normalization_factor
# ---------------------------------------------------------------------------

def test_get_psd_normalization_factor_standard():
    window = np.ones(100)
    fs = 100.0
    factor = get_psd_normalization_factor(window, fs, mode="standard")
    # Standard: 1 / (fs * sum(w^2)) = 1 / (100 * 100) = 1e-4
    assert factor == pytest.approx(1.0 / (fs * np.sum(window**2)))


def test_get_psd_normalization_factor_dtt():
    window = np.ones(100)
    fs = 100.0
    n = len(window)
    factor = get_psd_normalization_factor(window, fs, mode="dtt")
    # DTT: (sum_w^2) / (fs * N) = (N^2) / (fs * N) = N / fs
    assert factor == pytest.approx(n / fs)


def test_get_psd_normalization_factor_zero_window():
    window = np.zeros(50)
    assert get_psd_normalization_factor(window, fs=100.0) == 0.0


def test_get_psd_normalization_factor_hann():
    from scipy.signal import windows
    w = windows.hann(256)
    factor = get_psd_normalization_factor(w, fs=256.0, mode="standard")
    assert factor > 0


# ---------------------------------------------------------------------------
# convert_scipy_to_dtt — zero window passthrough
# ---------------------------------------------------------------------------

def test_convert_scipy_to_dtt_zero_window_passthrough():
    psd = np.array([1.0, 2.0, 3.0])
    window = np.zeros(10)
    result = convert_scipy_to_dtt(psd, window)
    np.testing.assert_array_equal(result, psd)


# ---------------------------------------------------------------------------
# get_window_normalized
# ---------------------------------------------------------------------------

def test_get_window_normalized_returns_array():
    w = get_window_normalized("hann", 64)
    assert isinstance(w, np.ndarray)
    assert len(w) == 64


def test_get_window_normalized_dtt_mode():
    w = get_window_normalized("hann", 128, mode="dtt")
    assert len(w) == 128
