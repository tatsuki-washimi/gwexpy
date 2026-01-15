"""Tests for gwexpy.noise.asd - ASD quantity/unit semantics.

This test module validates:
1. quantity parameter validation (allowed values, ValueError for invalid)
2. Unit correspondence for each quantity
3. Conversion correctness (strain <-> darm, acceleration -> velocity/displacement)
4. Edge cases (f=0 handling)
"""

import importlib
import sys
import types

import numpy as np
import pytest


def _install_gwinc_stub(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _install_obspy_stub(monkeypatch: pytest.MonkeyPatch) -> None:
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


class TestFromPygwincQuantityValidation:
    """Test from_pygwinc quantity parameter validation."""

    @pytest.fixture(autouse=True)
    def _gwinc_stub(self, monkeypatch):
        _install_gwinc_stub(monkeypatch)

    def test_strain_quantity_returns_frequencyseries(self):
        """from_pygwinc with quantity='strain' returns FrequencySeries."""
        from gwexpy.noise.asd import from_pygwinc
        from gwexpy.frequencyseries import FrequencySeries

        asd = from_pygwinc("aLIGO", fmin=10.0, fmax=100.0, df=10.0, quantity="strain")
        assert isinstance(asd, FrequencySeries)

    def test_strain_quantity_unit(self):
        """from_pygwinc with quantity='strain' has unit 1/sqrt(Hz)."""
        pytest.importorskip("astropy")
        from astropy import units as u
        from gwexpy.noise.asd import from_pygwinc

        asd = from_pygwinc("aLIGO", fmin=10.0, fmax=100.0, df=10.0, quantity="strain")
        expected_unit = u.Unit("1 / Hz^(1/2)")
        assert asd.unit.is_equivalent(expected_unit)

    def test_darm_quantity_returns_frequencyseries(self):
        """from_pygwinc with quantity='darm' returns FrequencySeries."""
        from gwexpy.noise.asd import from_pygwinc
        from gwexpy.frequencyseries import FrequencySeries

        asd = from_pygwinc("aLIGO", fmin=10.0, fmax=100.0, df=10.0, quantity="darm")
        assert isinstance(asd, FrequencySeries)

    def test_darm_quantity_unit(self):
        """from_pygwinc with quantity='darm' has unit m/sqrt(Hz)."""
        pytest.importorskip("astropy")
        from astropy import units as u
        from gwexpy.noise.asd import from_pygwinc

        asd = from_pygwinc("aLIGO", fmin=10.0, fmax=100.0, df=10.0, quantity="darm")
        expected_unit = u.Unit("m / Hz^(1/2)")
        assert asd.unit.is_equivalent(expected_unit)

    def test_displacement_alias_for_darm(self):
        """from_pygwinc with quantity='displacement' is alias for 'darm'."""
        from gwexpy.noise.asd import from_pygwinc

        freqs = np.arange(10.0, 101.0, 10.0)
        darm_asd = from_pygwinc("aLIGO", frequencies=freqs, quantity="darm")
        disp_asd = from_pygwinc("aLIGO", frequencies=freqs, quantity="displacement")

        # Same unit
        assert darm_asd.unit == disp_asd.unit
        # Same values
        np.testing.assert_allclose(darm_asd.value, disp_asd.value)

    def test_velocity_quantity_raises_valueerror(self):
        """from_pygwinc with quantity='velocity' raises ValueError."""
        from gwexpy.noise.asd import from_pygwinc

        with pytest.raises(ValueError, match="velocity"):
            from_pygwinc("aLIGO", quantity="velocity")

    def test_acceleration_quantity_raises_valueerror(self):
        """from_pygwinc with quantity='acceleration' raises ValueError."""
        from gwexpy.noise.asd import from_pygwinc

        with pytest.raises(ValueError, match="acceleration"):
            from_pygwinc("aLIGO", quantity="acceleration")

    def test_invalid_quantity_raises_valueerror(self):
        """from_pygwinc with invalid quantity raises ValueError."""
        from gwexpy.noise.asd import from_pygwinc

        with pytest.raises(ValueError):
            from_pygwinc("aLIGO", quantity="invalid")

    def test_darm_strain_ratio_equals_arm_length(self):
        """darm / strain ratio equals arm length L at all frequencies."""
        import gwinc
        from gwexpy.noise.asd import from_pygwinc

        freqs = np.arange(10.0, 101.0, 10.0)
        strain_asd = from_pygwinc("aLIGO", frequencies=freqs, quantity="strain")
        darm_asd = from_pygwinc("aLIGO", frequencies=freqs, quantity="darm")

        # Get arm length from IFO
        budget = gwinc.load_budget("aLIGO")
        arm_length = budget.ifo.Infrastructure.Length

        # Ratio should equal arm length
        ratio = darm_asd.value / strain_asd.value
        np.testing.assert_allclose(ratio, arm_length, rtol=1e-10)

    def test_fmin_greater_than_fmax_raises_valueerror(self):
        """from_pygwinc with fmin >= fmax raises ValueError."""
        from gwexpy.noise.asd import from_pygwinc

        with pytest.raises(ValueError, match="fmin"):
            from_pygwinc("aLIGO", fmin=100.0, fmax=10.0)


class TestFromObspyQuantityValidation:
    """Test from_obspy quantity parameter validation."""

    @pytest.fixture(autouse=True)
    def _obspy_stub(self, monkeypatch):
        _install_obspy_stub(monkeypatch)

    def test_acceleration_quantity_returns_frequencyseries(self):
        """from_obspy with quantity='acceleration' returns FrequencySeries."""
        from gwexpy.noise.asd import from_obspy
        from gwexpy.frequencyseries import FrequencySeries

        asd = from_obspy("NLNM", quantity="acceleration")
        assert isinstance(asd, FrequencySeries)

    def test_acceleration_quantity_unit(self):
        """from_obspy with quantity='acceleration' has unit m/(s²·sqrt(Hz))."""
        pytest.importorskip("astropy")
        from astropy import units as u
        from gwexpy.noise.asd import from_obspy

        asd = from_obspy("NLNM", quantity="acceleration")
        expected_unit = u.Unit("m / s^2 / Hz^(1/2)")
        assert asd.unit.is_equivalent(expected_unit)

    def test_velocity_quantity_unit(self):
        """from_obspy with quantity='velocity' has unit m/(s·sqrt(Hz))."""
        pytest.importorskip("astropy")
        from astropy import units as u
        from gwexpy.noise.asd import from_obspy

        asd = from_obspy("NLNM", quantity="velocity")
        expected_unit = u.Unit("m / s / Hz^(1/2)")
        assert asd.unit.is_equivalent(expected_unit)

    def test_displacement_quantity_unit(self):
        """from_obspy with quantity='displacement' has unit m/sqrt(Hz)."""
        pytest.importorskip("astropy")
        from astropy import units as u
        from gwexpy.noise.asd import from_obspy

        asd = from_obspy("NLNM", quantity="displacement")
        expected_unit = u.Unit("m / Hz^(1/2)")
        assert asd.unit.is_equivalent(expected_unit)

    def test_strain_quantity_raises_valueerror(self):
        """from_obspy with quantity='strain' raises ValueError."""
        from gwexpy.noise.asd import from_obspy

        with pytest.raises(ValueError, match="strain.*not supported"):
            from_obspy("NLNM", quantity="strain")

    def test_invalid_quantity_raises_valueerror(self):
        """from_obspy with invalid quantity raises ValueError."""
        from gwexpy.noise.asd import from_obspy

        with pytest.raises(ValueError):
            from_obspy("NLNM", quantity="invalid")

    def test_f0_conversion_produces_nan(self):
        """Conversion at f=0 produces NaN, not inf."""
        from gwexpy.noise.asd import from_obspy

        freqs = np.array([0.0, 0.01, 0.1, 1.0])

        # Velocity conversion
        vel_asd = from_obspy("NLNM", frequencies=freqs, quantity="velocity")
        assert np.isnan(vel_asd.value[0])
        assert not np.isinf(vel_asd.value[0])

        # Displacement conversion
        disp_asd = from_obspy("NLNM", frequencies=freqs, quantity="displacement")
        assert np.isnan(disp_asd.value[0])
        assert not np.isinf(disp_asd.value[0])

    def test_velocity_displacement_conversion_consistency(self):
        """Velocity and displacement conversions are consistent (v = d * 2πf)."""
        from gwexpy.noise.asd import from_obspy

        freqs = np.array([0.01, 0.1, 1.0, 10.0])
        vel_asd = from_obspy("NLNM", frequencies=freqs, quantity="velocity")
        disp_asd = from_obspy("NLNM", frequencies=freqs, quantity="displacement")

        # velocity = displacement * 2πf (in frequency domain, lower = more displacement)
        omega = 2 * np.pi * freqs
        expected_vel = disp_asd.value * omega
        np.testing.assert_allclose(vel_asd.value, expected_vel, rtol=1e-10)

    def test_unknown_model_raises_valueerror(self):
        """from_obspy with unknown model raises ValueError."""
        from gwexpy.noise.asd import from_obspy

        with pytest.raises(ValueError, match="Unknown model"):
            from_obspy("UNKNOWN_MODEL")

    def test_nhnm_model_works(self):
        """from_obspy with NHNM model returns valid FrequencySeries."""
        from gwexpy.noise.asd import from_obspy
        from gwexpy.frequencyseries import FrequencySeries

        asd = from_obspy("NHNM")
        assert isinstance(asd, FrequencySeries)
        assert asd.name == "NHNM"

    def test_infrasound_models_work(self):
        """from_obspy with infrasound models (IDCH, IDCL) works."""
        pytest.importorskip("astropy")
        from astropy import units as u
        from gwexpy.noise.asd import from_obspy
        from gwexpy.frequencyseries import FrequencySeries

        for model in ["IDCH", "IDCL"]:
            asd = from_obspy(model)
            assert isinstance(asd, FrequencySeries)
            assert asd.name == model
            # Infrasound models return Pa/sqrt(Hz)
            expected_unit = u.Unit("Pa / Hz^(1/2)")
            assert asd.unit.is_equivalent(expected_unit)
