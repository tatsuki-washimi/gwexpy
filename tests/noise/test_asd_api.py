"""Tests for gwexpy.noise ASD API using gwinc/obspy stubs."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def _install_gwinc_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("astropy")
    stub = types.ModuleType("gwinc")

    class StubBudget:
        def __init__(self, arm_length: float) -> None:
            self.ifo = types.SimpleNamespace(
                Infrastructure=types.SimpleNamespace(Length=arm_length)
            )

        def run(self, freq=None, **kwargs):
            if freq is None:
                freq = kwargs.get("freq")
            psd = np.full_like(np.asarray(freq, dtype=float), 4.0, dtype=float)
            return types.SimpleNamespace(psd=psd)

    def load_budget(model: str) -> StubBudget:
        return StubBudget(4000.0)

    stub.load_budget = load_budget
    monkeypatch.setitem(sys.modules, "gwinc", stub)

    import gwexpy.noise.gwinc_ as gwinc_

    importlib.reload(gwinc_)

    import gwexpy.noise.asd as asd

    importlib.reload(asd)


def _install_obspy_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("astropy")
    obspy_mod = types.ModuleType("obspy")
    signal_mod = types.ModuleType("obspy.signal")
    spectral_mod = types.ModuleType("obspy.signal.spectral_estimation")

    periods = np.array([1.0, 0.5])
    psd_db = np.array([-20.0, -20.0])

    def _model():
        return periods, psd_db

    spectral_mod.get_nhnm = _model
    spectral_mod.get_nlnm = _model
    spectral_mod.get_idc_infra_hi_noise = _model
    spectral_mod.get_idc_infra_low_noise = _model

    signal_mod.spectral_estimation = spectral_mod
    obspy_mod.signal = signal_mod

    monkeypatch.setitem(sys.modules, "obspy", obspy_mod)
    monkeypatch.setitem(sys.modules, "obspy.signal", signal_mod)
    monkeypatch.setitem(sys.modules, "obspy.signal.spectral_estimation", spectral_mod)

    import gwexpy.noise.obspy_ as obspy_

    importlib.reload(obspy_)

    import gwexpy.noise.asd as asd

    importlib.reload(asd)


class TestFromPygwincStub:
    def test_units_and_displacement_alias(self, monkeypatch):
        pytest.importorskip("astropy")
        _install_gwinc_stub(monkeypatch)

        from astropy import units as u

        from gwexpy.noise.asd import from_pygwinc

        freqs = np.array([10.0, 20.0, 30.0])
        strain = from_pygwinc("aLIGO", frequencies=freqs, quantity="strain")
        darm = from_pygwinc("aLIGO", frequencies=freqs, quantity="darm")
        disp = from_pygwinc("aLIGO", frequencies=freqs, quantity="displacement")

        assert strain.unit.is_equivalent(u.Unit("1 / Hz^(1/2)"))
        assert darm.unit.is_equivalent(u.Unit("m / Hz^(1/2)"))
        assert disp.unit == darm.unit
        np.testing.assert_allclose(disp.value, darm.value)
        np.testing.assert_allclose(darm.value / strain.value, 4000.0)

    def test_invalid_quantities_raise(self, monkeypatch):
        _install_gwinc_stub(monkeypatch)

        from gwexpy.noise.asd import from_pygwinc

        with pytest.raises(ValueError):
            from_pygwinc("aLIGO", quantity="velocity")
        with pytest.raises(ValueError):
            from_pygwinc("aLIGO", quantity="acceleration")
        with pytest.raises(ValueError):
            from_pygwinc("aLIGO", quantity="invalid")

    def test_fmin_greater_equal_fmax_raises(self, monkeypatch):
        _install_gwinc_stub(monkeypatch)

        from gwexpy.noise.asd import from_pygwinc

        with pytest.raises(ValueError):
            from_pygwinc("aLIGO", fmin=10.0, fmax=10.0)

    def test_reexports_from_asd_and_noise(self, monkeypatch):
        _install_gwinc_stub(monkeypatch)

        from gwexpy.noise import from_pygwinc as noise_from_pygwinc
        from gwexpy.noise.asd import from_pygwinc as asd_from_pygwinc

        freqs = np.array([10.0, 20.0])
        asd = asd_from_pygwinc("aLIGO", frequencies=freqs, quantity="strain")
        noise = noise_from_pygwinc("aLIGO", frequencies=freqs, quantity="strain")
        np.testing.assert_allclose(asd.value, noise.value)


class TestFromObspyStub:
    def test_strain_quantity_raises(self, monkeypatch):
        _install_obspy_stub(monkeypatch)

        from gwexpy.noise.asd import from_obspy

        with pytest.raises(ValueError):
            from_obspy("NLNM", quantity="strain")

    def test_unknown_model_raises(self, monkeypatch):
        _install_obspy_stub(monkeypatch)

        from gwexpy.noise.asd import from_obspy

        with pytest.raises(ValueError):
            from_obspy("UNKNOWN")

    def test_units_for_seismic_quantities(self, monkeypatch):
        pytest.importorskip("astropy")
        _install_obspy_stub(monkeypatch)

        from astropy import units as u

        from gwexpy.noise.asd import from_obspy

        acc = from_obspy("NLNM", quantity="acceleration")
        vel = from_obspy("NLNM", quantity="velocity")
        disp = from_obspy("NLNM", quantity="displacement")

        assert acc.unit.is_equivalent(u.Unit("m / s^2 / Hz^(1/2)"))
        assert vel.unit.is_equivalent(u.Unit("m / s / Hz^(1/2)"))
        assert disp.unit.is_equivalent(u.Unit("m / Hz^(1/2)"))

    def test_zero_frequency_converts_to_nan(self, monkeypatch):
        _install_obspy_stub(monkeypatch)

        from gwexpy.noise.asd import from_obspy

        freqs = np.array([0.0, 1.0, 2.0])
        vel = from_obspy("NLNM", frequencies=freqs, quantity="velocity")
        disp = from_obspy("NLNM", frequencies=freqs, quantity="displacement")

        assert np.isnan(vel.value[0])
        assert np.isnan(disp.value[0])
        assert not np.isinf(vel.value[0])
        assert not np.isinf(disp.value[0])

    def test_infrasound_ignores_quantity(self, monkeypatch):
        pytest.importorskip("astropy")
        _install_obspy_stub(monkeypatch)

        from astropy import units as u

        from gwexpy.noise.asd import from_obspy

        freqs = np.array([1.0, 2.0])
        asd = from_obspy("IDCH", frequencies=freqs, quantity="displacement")
        assert asd.unit.is_equivalent(u.Unit("Pa / Hz^(1/2)"))

    def test_reexports_from_asd_and_noise(self, monkeypatch):
        _install_obspy_stub(monkeypatch)

        from gwexpy.noise import from_obspy as noise_from_obspy
        from gwexpy.noise.asd import from_obspy as asd_from_obspy

        freqs = np.array([1.0, 2.0])
        asd = asd_from_obspy("NHNM", frequencies=freqs)
        noise = noise_from_obspy("NHNM", frequencies=freqs)
        np.testing.assert_allclose(asd.value, noise.value)
