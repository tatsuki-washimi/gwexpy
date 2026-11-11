"""Explicit window arrays in the overlap standard-error correction (#772)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u
from scipy.signal import get_window

from gwexpy.spectral import bootstrap_spectrogram, calculate_correlation_factor

NPERSEG = 256
NOVERLAP = 128
N_BLOCKS = 50


def _hann_factor():
    return calculate_correlation_factor("hann", NPERSEG, NOVERLAP, N_BLOCKS)


@pytest.mark.parametrize("container", [np.asarray, list], ids=["ndarray", "list"])
def test_valid_window_array_matches_named_window(container):
    window = container(get_window("hann", NPERSEG))

    factor = calculate_correlation_factor(window, NPERSEG, NOVERLAP, N_BLOCKS)

    assert factor > 1.0
    assert factor == pytest.approx(_hann_factor(), rel=1e-12)


@pytest.mark.parametrize("length", [NPERSEG + 5, NPERSEG - 1, 1])
@pytest.mark.parametrize("container", [np.asarray, list], ids=["ndarray", "list"])
def test_window_array_length_mismatch_is_rejected(container, length):
    window = container(get_window("hann", length))

    with pytest.raises(ValueError, match=rf"window.*{length}.*nperseg.*{NPERSEG}"):
        calculate_correlation_factor(window, NPERSEG, NOVERLAP, N_BLOCKS)


def test_window_array_must_be_one_dimensional():
    window = np.ones((NPERSEG, 2))

    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_correlation_factor(window, NPERSEG, NOVERLAP, N_BLOCKS)


def test_invalid_window_name_still_warns_and_falls_back():
    with pytest.warns(UserWarning, match="Could not generate window"):
        factor = calculate_correlation_factor(
            "no-such-window", NPERSEG, NOVERLAP, N_BLOCKS
        )

    assert factor == 1.0


def test_no_correction_requested_ignores_window_length():
    """Without overlap information there is nothing to validate against."""
    window = get_window("hann", NPERSEG + 5)

    assert calculate_correlation_factor(window, None, None, N_BLOCKS) == 1.0
    assert calculate_correlation_factor(window, NPERSEG, NOVERLAP, 1) == 1.0


class _DummySpectrogram:
    def __init__(self, value, *, dt, df, frequencies):
        self.value = np.asarray(value)
        self.dt = dt
        self.df = df
        self.frequencies = frequencies
        self.unit = "1"
        self.name = "dummy"


def _spectrogram(n_time=64, n_freq=16):
    rng = np.random.default_rng(772)
    data = rng.lognormal(mean=0.0, sigma=0.5, size=(n_time, n_freq))
    return _DummySpectrogram(
        data,
        dt=1.0 * u.s,
        df=1.0 * u.Hz,
        frequencies=np.linspace(10, 1000, n_freq) * u.Hz,
    )


def _bootstrap(window, overlap):
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return bootstrap_spectrogram(
            _spectrogram(),
            n_boot=100,
            average="mean",
            ci=0.68,
            window=window,
            fftlength=256.0,
            overlap=overlap,
        )


@pytest.mark.parametrize("length", [256, 1000, 333])
def test_bootstrap_window_array_gets_the_overlap_correction(length):
    """A window array must not silently lose the overlap correction."""
    window = get_window("hann", length)

    base = _bootstrap(window, 0.0).error_low.value
    by_array = _bootstrap(window, 128.0).error_low.value
    by_name = _bootstrap("hann", 128.0).error_low.value

    mask = base > 0
    ratio = by_array[mask] / base[mask]
    assert np.all(ratio > 1.01)
    np.testing.assert_allclose(by_array, by_name, rtol=1e-3)


def test_bootstrap_named_window_result_is_unchanged():
    spec_factor = calculate_correlation_factor("hann", 1000, 500, 64)

    base = _bootstrap("hann", 0.0).error_low.value
    scaled = _bootstrap("hann", 128.0).error_low.value

    mask = base > 0
    np.testing.assert_allclose(scaled[mask] / base[mask], spec_factor, rtol=1e-12)


def _bootstrap_from_metadata(window, dt):
    spec = _spectrogram()
    spec.dt = dt * u.s
    np.random.seed(0)
    return bootstrap_spectrogram(
        spec, n_boot=100, average="mean", ci=0.68, window=window
    )


@pytest.mark.parametrize("length", [256, 333])
def test_bootstrap_window_array_correction_from_spectrogram_metadata(length):
    """Without ``fftlength`` the overlap is inferred from ``dt`` and ``df``."""
    window = get_window("hann", length)

    base = _bootstrap_from_metadata(window, 1.0).error_low.value
    by_array = _bootstrap_from_metadata(window, 0.5).error_low.value
    by_name = _bootstrap_from_metadata("hann", 0.5).error_low.value

    mask = base > 0
    assert np.all(by_array[mask] / base[mask] > 1.01)
    np.testing.assert_allclose(by_array, by_name, rtol=1e-3)


def test_bootstrap_window_array_with_default_overlap():
    window = get_window("hann", 256)

    by_array = _bootstrap(window, None).error_low.value
    by_name = _bootstrap("hann", None).error_low.value

    np.testing.assert_allclose(by_array, by_name, rtol=1e-3)
