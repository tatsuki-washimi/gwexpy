"""Tests for gwexpy/noise/gwinc_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.noise.gwinc_ import from_pygwinc


def _fake_gwinc(arm_length=4000.0):
    """Return a minimal fake gwinc module."""
    class FakeTrace:
        psd = np.ones(10) * 1e-46  # strain^2/Hz

    class FakeBudget:
        def run(self, freq):
            return FakeTrace()

        @property
        def ifo(self):
            infra = SimpleNamespace(Length=arm_length)
            return SimpleNamespace(Infrastructure=infra)

    return SimpleNamespace(
        load_budget=lambda model: FakeBudget()
    )


class TestFromPygwincValidation:
    def test_raises_import_error_without_gwinc(self):
        with patch.dict(sys.modules, {"gwinc": None}):
            with pytest.raises(ImportError, match="pygwinc"):
                from_pygwinc("aLIGO")

    def test_invalid_quantity_raises(self):
        fake_gwinc = _fake_gwinc()
        with patch.dict(sys.modules, {"gwinc": fake_gwinc}):
            with pytest.raises(ValueError, match="Invalid quantity"):
                from_pygwinc("aLIGO", quantity="velocity")

    def test_fmin_greater_than_fmax_raises(self):
        fake_gwinc = _fake_gwinc()
        with patch.dict(sys.modules, {"gwinc": fake_gwinc}):
            with pytest.raises(ValueError, match="fmin"):
                from_pygwinc("aLIGO", fmin=1000.0, fmax=10.0)

    def test_fmin_equal_fmax_raises(self):
        fake_gwinc = _fake_gwinc()
        with patch.dict(sys.modules, {"gwinc": fake_gwinc}):
            with pytest.raises(ValueError, match="fmin"):
                from_pygwinc("aLIGO", fmin=100.0, fmax=100.0)


class TestFromPygwincModels:
    def test_strain_quantity(self):
        fake_gwinc = _fake_gwinc()
        with patch.dict(sys.modules, {"gwinc": fake_gwinc}):
            fs = from_pygwinc("aLIGO", fmin=10.0, fmax=100.0, df=10.0)
        assert isinstance(fs, FrequencySeries)
        assert fs.name == "aLIGO"

    def test_darm_quantity(self):
        fake_gwinc = _fake_gwinc(arm_length=4000.0)
        with patch.dict(sys.modules, {"gwinc": fake_gwinc}):
            fs = from_pygwinc("aLIGO", fmin=10.0, fmax=100.0, df=10.0, quantity="darm")
        assert isinstance(fs, FrequencySeries)
        # darm = strain * arm_length → values should be larger
        assert np.all(fs.value > 0)

    def test_displacement_alias_for_darm(self):
        fake_gwinc = _fake_gwinc()
        with patch.dict(sys.modules, {"gwinc": fake_gwinc}):
            fs = from_pygwinc("aLIGO", fmin=10.0, fmax=100.0, df=10.0, quantity="displacement")
        assert isinstance(fs, FrequencySeries)

    def test_aplus_model_alias(self):
        fake_gwinc = _fake_gwinc()
        with patch.dict(sys.modules, {"gwinc": fake_gwinc}):
            fs = from_pygwinc("A+", fmin=10.0, fmax=100.0, df=10.0)
        assert fs.name == "Aplus"

    def test_custom_frequencies(self):
        fake_gwinc = _fake_gwinc()
        freqs = np.array([10.0, 50.0, 100.0, 500.0])

        class FakeTraceCustom:
            psd = np.ones(4) * 1e-46

        class FakeBudgetCustom:
            def run(self, freq):
                return FakeTraceCustom()

            @property
            def ifo(self):
                return SimpleNamespace(Infrastructure=SimpleNamespace(Length=4000.0))

        fake_gwinc_custom = SimpleNamespace(
            load_budget=lambda m: FakeBudgetCustom()
        )
        with patch.dict(sys.modules, {"gwinc": fake_gwinc_custom}):
            fs = from_pygwinc("aLIGO", frequencies=freqs)
        assert len(fs) == 4

    def test_darm_missing_arm_length_raises(self):
        class FakeTrace:
            psd = np.ones(5) * 1e-46

        class FakeBudgetNoIFO:
            def run(self, freq):
                return FakeTrace()

            @property
            def ifo(self):
                # Missing Infrastructure.Length
                return SimpleNamespace(Infrastructure=SimpleNamespace())

        fake_gwinc_bad = SimpleNamespace(load_budget=lambda m: FakeBudgetNoIFO())
        with patch.dict(sys.modules, {"gwinc": fake_gwinc_bad}):
            with pytest.raises(ValueError, match="arm length"):
                from_pygwinc("aLIGO", fmin=10.0, fmax=100.0, df=10.0, quantity="darm")

    def test_default_frequency_array_generated(self):
        fake_gwinc = _fake_gwinc()

        class FakeTraceSize:
            def __init__(self, n):
                self.psd = np.ones(n) * 1e-46

        class FakeBudgetSized:
            def run(self, freq):
                return FakeTraceSize(len(freq))

            @property
            def ifo(self):
                return SimpleNamespace(Infrastructure=SimpleNamespace(Length=4000.0))

        fake_gwinc_sized = SimpleNamespace(load_budget=lambda m: FakeBudgetSized())
        with patch.dict(sys.modules, {"gwinc": fake_gwinc_sized}):
            fs = from_pygwinc("aLIGO", fmin=10.0, fmax=20.0, df=5.0)
        # np.arange(10, 25, 5) = [10, 15, 20] → 3 elements
        assert len(fs) == 3
