"""Tests for multitaper spectral estimation interoperability.

Uses mock objects and synthetic numpy arrays to simulate MTSpec / MTSine /
mtspec output.  Does NOT require multitaper or mtspec to be installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.interop.multitaper_ import from_mtspec, from_mtspec_array

N_FREQ = 512
FS = 1024.0  # sample rate [Hz]
FREQ_RES = FS / (2 * N_FREQ)  # 1.0 Hz


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_freq():
    """Linearly spaced frequency axis 0 … FS/2."""
    return np.linspace(0.0, FS / 2, N_FREQ)


def _make_psd(freq, f0=100.0, width=5.0, noise_level=1e-10):
    """Gaussian peak PSD centred at f0."""
    rng = np.random.default_rng(42)
    psd = noise_level * np.ones(len(freq))
    psd += np.exp(-0.5 * ((freq - f0) / width) ** 2)
    return psd


def _make_mtspec_obj(with_ci=True, with_se=True):
    """Simulate a Prieto MTSpec object (SimpleNamespace)."""
    freq = _make_freq()
    spec = _make_psd(freq)

    obj = SimpleNamespace(freq=freq, spec=spec, nw=4.0, kspec=7)

    if with_ci:
        ci = np.column_stack([spec * 0.8, spec * 1.2])  # (nf, 2)
        obj.spec_ci = ci

    if with_se:
        obj.se = np.full(N_FREQ, 14.0)  # 2 * kspec degrees of freedom

    return obj


def _make_mtsine_obj():
    """Simulate a Prieto MTSine object (no spec_ci, has kuse)."""
    freq = _make_freq()
    spec = _make_psd(freq)
    return SimpleNamespace(freq=freq, spec=spec, kuse=np.full(N_FREQ, 5.0))


# ---------------------------------------------------------------------------
# from_mtspec — Prieto interface
# ---------------------------------------------------------------------------


class TestFromMtspec:
    def test_returns_fs_no_ci(self):
        mt = _make_mtspec_obj(with_ci=False)
        result = from_mtspec(FrequencySeries, mt)
        assert isinstance(result, FrequencySeries)

    def test_returns_fs_when_include_ci_false(self):
        mt = _make_mtspec_obj(with_ci=True)
        result = from_mtspec(FrequencySeries, mt, include_ci=False)
        assert isinstance(result, FrequencySeries)

    def test_returns_dict_with_ci(self):
        mt = _make_mtspec_obj(with_ci=True)
        result = from_mtspec(FrequencySeriesDict, mt, include_ci=True)
        assert isinstance(result, FrequencySeriesDict)
        assert "psd" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_dict_has_three_entries(self):
        mt = _make_mtspec_obj(with_ci=True)
        result = from_mtspec(FrequencySeriesDict, mt)
        assert len(result) == 3

    def test_psd_length(self):
        mt = _make_mtspec_obj(with_ci=False)
        result = from_mtspec(FrequencySeries, mt)
        assert len(result) == N_FREQ

    def test_asd_is_sqrt_of_psd(self):
        mt = _make_mtspec_obj(with_ci=False)
        psd_result = from_mtspec(FrequencySeries, mt, quantity="psd")
        asd_result = from_mtspec(FrequencySeries, mt, quantity="asd")
        np.testing.assert_allclose(
            np.asarray(asd_result.value),
            np.sqrt(np.asarray(psd_result.value)),
            rtol=1e-10,
        )

    def test_asd_ci_is_sqrt_of_psd_ci(self):
        mt = _make_mtspec_obj(with_ci=True)
        psd_dict = from_mtspec(FrequencySeriesDict, mt, quantity="psd")
        asd_dict = from_mtspec(FrequencySeriesDict, mt, quantity="asd")
        np.testing.assert_allclose(
            np.asarray(asd_dict["ci_lower"].value),
            np.sqrt(np.asarray(psd_dict["ci_lower"].value)),
            rtol=1e-10,
        )

    def test_peak_frequency_preserved(self):
        mt = _make_mtspec_obj(with_ci=False)
        result = from_mtspec(FrequencySeries, mt)
        peak_idx = int(np.argmax(np.asarray(result.value)))
        freq_arr = _make_freq()
        assert abs(freq_arr[peak_idx] - 100.0) < 5.0  # within 5 Hz of 100 Hz

    def test_metadata_nw_stored(self):
        mt = _make_mtspec_obj(with_ci=False)
        result = from_mtspec(FrequencySeries, mt)
        # Metadata stored as _mt_* instance attributes
        assert hasattr(result, "_mt_nw")
        assert result._mt_nw == pytest.approx(4.0)

    def test_invalid_quantity_raises(self):
        mt = _make_mtspec_obj(with_ci=False)
        with pytest.raises(ValueError, match="quantity must be"):
            from_mtspec(FrequencySeries, mt, quantity="invalid")

    def test_empty_freq_raises(self):
        mt = SimpleNamespace(freq=np.array([]), spec=np.array([]))
        with pytest.raises(ValueError, match="empty frequency axis"):
            from_mtspec(FrequencySeries, mt)

    def test_mtsine_no_ci_returns_fs(self):
        mt = _make_mtsine_obj()
        result = from_mtspec(FrequencySeries, mt, include_ci=True)
        # MTSine has no spec_ci → must return FrequencySeries, not dict
        assert isinstance(result, FrequencySeries)

    def test_ci_lower_le_main_le_upper(self):
        mt = _make_mtspec_obj(with_ci=True)
        result = from_mtspec(FrequencySeriesDict, mt, quantity="psd")
        lo = np.asarray(result["ci_lower"].value)
        hi = np.asarray(result["ci_upper"].value)
        main = np.asarray(result["psd"].value)
        assert np.all(lo <= main + 1e-30)
        assert np.all(main <= hi + 1e-30)


# ---------------------------------------------------------------------------
# from_mtspec_array — Krischer / array interface
# ---------------------------------------------------------------------------


class TestFromMtspecArray:
    def test_returns_fs(self):
        freq = _make_freq()
        spec = _make_psd(freq)
        result = from_mtspec_array(FrequencySeries, spec, freq)
        assert isinstance(result, FrequencySeries)

    def test_length(self):
        freq = _make_freq()
        spec = _make_psd(freq)
        result = from_mtspec_array(FrequencySeries, spec, freq)
        assert len(result) == N_FREQ

    def test_returns_dict_with_ci(self):
        freq = _make_freq()
        spec = _make_psd(freq)
        ci_lo = spec * 0.8
        ci_hi = spec * 1.2
        result = from_mtspec_array(
            FrequencySeriesDict, spec, freq, ci_lower=ci_lo, ci_upper=ci_hi
        )
        assert isinstance(result, FrequencySeriesDict)
        assert "psd" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_asd_label(self):
        freq = _make_freq()
        spec = _make_psd(freq)
        result = from_mtspec_array(FrequencySeries, spec, freq, quantity="asd")
        assert isinstance(result, FrequencySeries)

    def test_unit_preserved(self):
        freq = _make_freq()
        spec = _make_psd(freq)
        result = from_mtspec_array(FrequencySeries, spec, freq, unit="m^2 / Hz")
        assert result.unit is not None

    def test_shape_mismatch_raises(self):
        freq = _make_freq()
        spec = _make_psd(freq)[:-1]  # one sample shorter
        with pytest.raises(ValueError, match="freq length"):
            from_mtspec_array(FrequencySeries, spec, freq)

    def test_empty_freq_raises(self):
        with pytest.raises(ValueError, match="freq array is empty"):
            from_mtspec_array(FrequencySeries, np.array([]), np.array([]))

    def test_invalid_quantity_raises(self):
        freq = _make_freq()
        spec = _make_psd(freq)
        with pytest.raises(ValueError, match="quantity must be"):
            from_mtspec_array(FrequencySeries, spec, freq, quantity="rms")

    def test_nonuniform_freq_raises(self):
        freq = np.array([0.0, 1.0, 2.0, 4.0, 5.0])  # gap at index 3
        spec = np.ones(5)
        with pytest.raises(ValueError, match="equally spaced"):
            from_mtspec_array(FrequencySeries, spec, freq)
