"""Tests for gwexpy/interop/pyspeckit_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries


def _make_fs(n=8, f0=1.0, df=1.0, unit="m"):
    return FrequencySeries(np.ones(n), f0=f0, df=df, unit=unit)


class TestToPyspeckit:
    def test_raises_without_pyspeckit(self):
        with patch.dict(sys.modules, {"pyspeckit": None}):
            from gwexpy.interop.pyspeckit_ import to_pyspeckit
            with pytest.raises(ImportError):
                to_pyspeckit(_make_fs())

    def test_no_frequencies_raises(self):
        fake_pyspeckit = SimpleNamespace(Spectrum=lambda **kw: kw)
        obj = SimpleNamespace(value=np.ones(5), unit=u.m)  # no .frequencies
        with patch.dict(sys.modules, {"pyspeckit": fake_pyspeckit}):
            from gwexpy.interop import pyspeckit_
            import importlib
            importlib.reload(pyspeckit_)
            with pytest.raises(ValueError, match="frequencies"):
                pyspeckit_.to_pyspeckit(obj)

    def test_basic_conversion(self):
        calls = {}
        def fake_spectrum(data, xarr, **kw):
            calls["data"] = data
            calls["xarr"] = xarr
            return SimpleNamespace(data=data, xarr=xarr)

        fake_pyspeckit = SimpleNamespace(Spectrum=fake_spectrum)
        with patch.dict(sys.modules, {"pyspeckit": fake_pyspeckit}):
            from gwexpy.interop import pyspeckit_
            import importlib
            importlib.reload(pyspeckit_)
            fs = _make_fs(n=6)
            result = pyspeckit_.to_pyspeckit(fs)
        assert calls["data"] is not None
        np.testing.assert_array_equal(calls["xarr"], fs.frequencies.value)


class TestFromPyspeckit:
    def test_basic_conversion(self):
        from gwexpy.interop.pyspeckit_ import from_pyspeckit
        freqs = np.array([1.0, 2.0, 3.0, 4.0])
        fake_spec = SimpleNamespace(
            data=np.ones(4),
            xarr=freqs,
        )
        fs = from_pyspeckit(FrequencySeries, fake_spec)
        assert fs is not None
        assert len(fs) == 4

    def test_passes_kwargs(self):
        from gwexpy.interop.pyspeckit_ import from_pyspeckit
        fake_spec = SimpleNamespace(
            data=np.ones(3),
            xarr=np.array([1.0, 2.0, 3.0]),
        )
        fs = from_pyspeckit(FrequencySeries, fake_spec, name="test")
        assert fs.name == "test"
