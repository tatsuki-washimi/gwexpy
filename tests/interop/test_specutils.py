"""Tests for gwexpy/interop/specutils_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries


def _make_fs(n=8, f0=1.0, df=1.0, unit="m"):
    return FrequencySeries(np.ones(n), f0=f0, df=df, unit=unit)


def _fake_specutils_and_units():
    """Fake specutils + astropy.units for mocking."""
    class FakeSpectrum1D:
        def __init__(self, flux, spectral_axis, **kwargs):
            self.flux = flux
            self.spectral_axis = spectral_axis

    fake_spec = SimpleNamespace(Spectrum1D=FakeSpectrum1D)
    return fake_spec


class TestToSpecutilsRequiresPackage:
    def test_raises_import_error_without_specutils(self):
        with patch.dict(sys.modules, {"specutils": None}):
            from gwexpy.interop.specutils_ import to_specutils
            with pytest.raises(ImportError):
                to_specutils(_make_fs())


class TestToSpecutilsWithMock:
    def test_basic_conversion(self):
        fake_spec = _fake_specutils_and_units()
        with patch.dict(sys.modules, {"specutils": fake_spec}):
            from gwexpy.interop import specutils_
            import importlib
            importlib.reload(specutils_)
            fs = _make_fs(n=5)
            result = specutils_.to_specutils(fs)
        assert result is not None

    def test_no_frequencies_raises(self):
        fake_spec = _fake_specutils_and_units()
        obj = SimpleNamespace(value=np.ones(5), unit=u.m)  # no .frequencies attr
        with patch.dict(sys.modules, {"specutils": fake_spec}):
            from gwexpy.interop import specutils_
            import importlib
            importlib.reload(specutils_)
            with pytest.raises(ValueError, match="frequencies"):
                specutils_.to_specutils(obj)

    def test_unit_fallback_when_multiplication_fails(self):
        fake_spec = _fake_specutils_and_units()

        class BadUnit:
            def __str__(self): return "bad"
            def __mul__(self, other): raise TypeError("can't multiply")
            def __rmul__(self, other): raise TypeError("can't multiply")

        fs = _make_fs(n=4)
        fs_mock = SimpleNamespace(
            value=np.ones(4),
            unit=BadUnit(),
            frequencies=fs.frequencies,
        )
        with patch.dict(sys.modules, {"specutils": fake_spec}):
            from gwexpy.interop import specutils_
            import importlib
            importlib.reload(specutils_)
            result = specutils_.to_specutils(fs_mock)
        assert result is not None

    def test_no_unit_uses_dimensionless(self):
        fake_spec = _fake_specutils_and_units()
        fs = _make_fs(n=4)
        fs_mock = SimpleNamespace(
            value=np.ones(4),
            unit=None,
            frequencies=fs.frequencies,
        )
        with patch.dict(sys.modules, {"specutils": fake_spec}):
            from gwexpy.interop import specutils_
            import importlib
            importlib.reload(specutils_)
            result = specutils_.to_specutils(fs_mock)
        assert result is not None

    def test_frequencies_without_unit(self):
        fake_spec = _fake_specutils_and_units()
        # frequencies without .unit attribute
        fake_freqs = SimpleNamespace(value=np.array([1.0, 2.0, 3.0, 4.0]))
        fs_mock = SimpleNamespace(
            value=np.ones(4),
            unit=u.m,
            frequencies=fake_freqs,
        )
        with patch.dict(sys.modules, {"specutils": fake_spec}):
            from gwexpy.interop import specutils_
            import importlib
            importlib.reload(specutils_)
            result = specutils_.to_specutils(fs_mock)
        assert result is not None


class TestFromSpecutils:
    def test_basic_conversion(self):
        from gwexpy.interop.specutils_ import from_specutils
        flux = np.ones(5) * u.m
        freqs = np.arange(1, 6) * u.Hz
        fake_spectrum = SimpleNamespace(flux=flux, spectral_axis=freqs)
        fs = from_specutils(FrequencySeries, fake_spectrum)
        assert fs is not None
        assert len(fs) == 5

    def test_passes_kwargs(self):
        from gwexpy.interop.specutils_ import from_specutils
        flux = np.ones(3)
        freqs = np.array([1.0, 2.0, 3.0]) * u.Hz
        fake_spectrum = SimpleNamespace(flux=flux, spectral_axis=freqs)
        fs = from_specutils(FrequencySeries, fake_spectrum, name="test_spec")
        assert fs.name == "test_spec"
