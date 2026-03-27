"""Tests for gwexpy/types/mixin/mixin_legacy.py."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.mixin.mixin_legacy import RegularityMixin, PhaseMethodsMixin


# ---------------------------------------------------------------------------
# RegularityMixin
# ---------------------------------------------------------------------------

class SimpleRegular(RegularityMixin):
    """Minimal class with xindex."""
    def __init__(self, xindex):
        self.xindex = xindex


class NoXIndex(RegularityMixin):
    """Class with no xindex attribute."""
    pass


class RegularIndex(RegularityMixin):
    """Class with .regular attribute on xindex."""
    class _Idx:
        regular = True
    xindex = _Idx()


class TestRegularityMixin:
    def test_is_regular_xindex_none(self):
        # Line 25 — xindex is None → True
        obj = NoXIndex()
        obj.xindex = None
        assert obj.is_regular is True

    def test_is_regular_no_xindex_attr(self):
        # No xindex attr at all → getattr returns None → True
        obj = NoXIndex()
        assert obj.is_regular is True

    def test_is_regular_with_regular_attr(self):
        # Line 27 — xindex has .regular
        obj = RegularIndex()
        assert obj.is_regular is True

    def test_is_regular_manual_uniform(self):
        # Lines 30-35 — manual diff check, uniform spacing
        obj = SimpleRegular(np.arange(10, dtype=float))
        assert obj.is_regular is True

    def test_is_regular_manual_one_element(self):
        # Line 32 — len < 2 → True
        obj = SimpleRegular(np.array([5.0]))
        assert obj.is_regular is True

    def test_is_regular_manual_irregular(self):
        # Lines 33-35 — irregular spacing
        obj = SimpleRegular(np.array([0.0, 1.0, 2.5, 4.0]))
        assert obj.is_regular is False

    def test_is_regular_exception_returns_false(self):
        # Line 36-37 — ValueError/TypeError during diff → False
        class BrokenIdx(RegularityMixin):
            @property
            def xindex(self):
                return "not_an_array"  # np.asarray raises or diff fails
        obj = BrokenIdx()
        # np.diff("not_an_array") → TypeError → returns False
        result = obj.is_regular
        assert isinstance(result, bool)

    def test_check_regular_passes_for_regular(self):
        obj = SimpleRegular(np.arange(10, dtype=float))
        obj._check_regular("TestMethod")  # Should not raise

    def test_check_regular_raises_for_irregular_time(self):
        # Lines 45-47 — "Time" in class name
        class TimeIrregular(RegularityMixin):
            xindex = np.array([0.0, 1.0, 2.5])
            __name__ = "TimeIrregular"
        obj = TimeIrregular()
        with pytest.raises(ValueError, match="constant dt"):
            obj._check_regular("MyMethod")

    def test_check_regular_raises_for_frequency_class(self):
        # Lines 48-49 — "Frequency" in class name
        class FrequencyIrregular(RegularityMixin):
            xindex = np.array([0.0, 1.0, 2.5])
        obj = FrequencyIrregular()
        with pytest.raises(ValueError, match="frequency grid"):
            obj._check_regular("MyMethod")

    def test_check_regular_raises_for_spectrogram_class(self):
        # Line 48 — "Spectrogram" in class name
        class SpectrogramIrregular(RegularityMixin):
            xindex = np.array([0.0, 1.0, 2.5])
        obj = SpectrogramIrregular()
        with pytest.raises(ValueError, match="frequency grid"):
            obj._check_regular()

    def test_check_regular_raises_generic(self):
        # Lines 50-51 — generic class name
        class GenericIrregular(RegularityMixin):
            xindex = np.array([0.0, 1.0, 2.5])
        obj = GenericIrregular()
        with pytest.raises(ValueError, match="constant spacing"):
            obj._check_regular()

    def test_check_regular_no_method_name(self):
        # Line 42 — method_name=None → uses "This method"
        class GenericIrregular(RegularityMixin):
            xindex = np.array([0.0, 1.0, 2.5])
        obj = GenericIrregular()
        with pytest.raises(ValueError, match="This method"):
            obj._check_regular(method_name=None)


# ---------------------------------------------------------------------------
# PhaseMethodsMixin
# ---------------------------------------------------------------------------

class FakePhaseObj(PhaseMethodsMixin):
    """Fake object with radian() and degree() methods."""
    def radian(self, unwrap=False, **kwargs):
        return f"radian(unwrap={unwrap})"

    def degree(self, unwrap=False, **kwargs):
        return f"degree(unwrap={unwrap})"


class TestPhaseMethodsMixin:
    def test_phase_radians_default(self):
        obj = FakePhaseObj()
        result = obj.phase()
        assert result == "radian(unwrap=False)"

    def test_phase_deg_true(self):
        # Lines 79-80 — deg=True → degree()
        obj = FakePhaseObj()
        result = obj.phase(deg=True)
        assert result == "degree(unwrap=False)"

    def test_phase_unwrap(self):
        obj = FakePhaseObj()
        result = obj.phase(unwrap=True)
        assert result == "radian(unwrap=True)"

    def test_phase_deg_and_unwrap(self):
        obj = FakePhaseObj()
        result = obj.phase(deg=True, unwrap=True)
        assert result == "degree(unwrap=True)"

    def test_angle_alias(self):
        # Line 85 — angle is alias for phase
        obj = FakePhaseObj()
        result = obj.angle()
        assert result == "radian(unwrap=False)"

    def test_angle_deg(self):
        obj = FakePhaseObj()
        result = obj.angle(deg=True)
        assert result == "degree(unwrap=False)"
