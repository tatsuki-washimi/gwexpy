"""Tests for gwexpy/noise/obspy_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.noise.obspy_ import _convert_seismic_quantity, from_obspy


def _fake_obspy_spectral_estimation():
    """Return a fake obspy.signal.spectral_estimation module."""
    # Periods in seconds, psd_db in dB
    periods = np.array([1.0, 0.5, 0.25, 0.1])  # freqs = [1, 2, 4, 10] Hz
    psd_db = np.array([-150.0, -155.0, -160.0, -165.0])

    return SimpleNamespace(
        get_nhnm=lambda: (periods.copy(), psd_db.copy()),
        get_nlnm=lambda: (periods.copy(), psd_db.copy()),
        get_idc_infra_hi_noise=lambda: (periods.copy(), psd_db.copy()),
        get_idc_infra_low_noise=lambda: (periods.copy(), psd_db.copy()),
    )


def _patch_obspy(fake_spec):
    fake_signal = SimpleNamespace(spectral_estimation=fake_spec)
    fake_obspy = SimpleNamespace(signal=fake_signal)
    return {
        "obspy": fake_obspy,
        "obspy.signal": fake_signal,
        "obspy.signal.spectral_estimation": fake_spec,
    }


class TestFromObspyValidation:
    def test_raises_import_error_without_obspy(self):
        with patch.dict(sys.modules, {"obspy": None, "obspy.signal": None,
                                       "obspy.signal.spectral_estimation": None}):
            with pytest.raises(ImportError, match="obspy"):
                from_obspy("NLNM")

    def test_strain_quantity_raises(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            with pytest.raises(ValueError, match="strain"):
                from_obspy("NLNM", quantity="strain")

    def test_invalid_quantity_raises(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            with pytest.raises(ValueError, match="Invalid quantity"):
                from_obspy("NLNM", quantity="jerk")

    def test_unknown_model_raises(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            with pytest.raises(ValueError, match="Unknown model"):
                from_obspy("XYZZY")


class TestFromObspyModels:
    def test_nhnm_acceleration(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            fs = from_obspy("NHNM", quantity="acceleration")
        assert isinstance(fs, FrequencySeries)
        assert fs.name == "NHNM"
        assert len(fs) > 0

    def test_nlnm_acceleration(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            fs = from_obspy("NLNM")
        assert isinstance(fs, FrequencySeries)

    def test_idch_model(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            fs = from_obspy("IDCH")
        assert isinstance(fs, FrequencySeries)

    def test_idcl_model(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            fs = from_obspy("IDCL")
        assert isinstance(fs, FrequencySeries)

    def test_lowercase_model_name(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            fs = from_obspy("nlnm")
        assert fs.name == "NLNM"

    def test_velocity_quantity(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            fs = from_obspy("NLNM", quantity="velocity")
        assert isinstance(fs, FrequencySeries)

    def test_displacement_quantity(self):
        fake_spec = _fake_obspy_spectral_estimation()
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            fs = from_obspy("NLNM", quantity="displacement")
        assert isinstance(fs, FrequencySeries)

    def test_with_custom_frequencies(self):
        fake_spec = _fake_obspy_spectral_estimation()
        freqs = np.array([1.0, 2.0, 5.0])
        with patch.dict(sys.modules, _patch_obspy(fake_spec)):
            fs = from_obspy("NLNM", frequencies=freqs)
        assert len(fs) == 3


class TestConvertSeismicQuantity:
    def _make_accel_fs(self, n=5):
        freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = np.ones(n) * 1e-9
        return FrequencySeries(data, frequencies=freqs, unit="m / s^2 / Hz^(1/2)")

    def test_velocity_conversion(self):
        fs = self._make_accel_fs()
        result = _convert_seismic_quantity(fs, "velocity")
        assert isinstance(result, FrequencySeries)
        # velocity = accel / (2*pi*f) → should be smaller
        assert np.all(np.isfinite(result.value))

    def test_displacement_conversion(self):
        fs = self._make_accel_fs()
        result = _convert_seismic_quantity(fs, "displacement")
        assert isinstance(result, FrequencySeries)
        assert np.all(np.isfinite(result.value))

    def test_f0_gives_nan(self):
        freqs = np.array([0.0, 1.0, 2.0])
        data = np.ones(3) * 1e-9
        fs = FrequencySeries(data, frequencies=freqs, unit="m / s^2 / Hz^(1/2)")
        result = _convert_seismic_quantity(fs, "velocity")
        assert np.isnan(result.value[0])
        assert np.all(np.isfinite(result.value[1:]))

    def test_unknown_quantity_raises(self):
        fs = self._make_accel_fs()
        with pytest.raises(ValueError, match="Unknown quantity"):
            _convert_seismic_quantity(fs, "jerk")
