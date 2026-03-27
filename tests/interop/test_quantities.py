"""Tests for gwexpy/interop/quantities_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


def _fake_pq():
    """Return a minimal fake quantities module."""
    class FakeQuantity:
        # High priority so NumPy defers to __rmul__ instead of trying to broadcast
        __array_priority__ = 20.0

        def __init__(self, val, unit_str=""):
            self._val = np.asarray(val, dtype=float)
            self._unit = unit_str

        def __mul__(self, other):
            if isinstance(other, FakeQuantity):
                return FakeQuantity(self._val * other._val)
            return FakeQuantity(self._val * np.asarray(other))

        def __rmul__(self, other):
            return FakeQuantity(np.asarray(other) * self._val, self._unit)

        @property
        def magnitude(self):
            return self._val

        @property
        def dimensionality(self):
            return SimpleNamespace(string=self._unit or "dimensionless")

        @property
        def units(self):
            return self._unit

        @units.setter
        def units(self, v):
            self._unit = v

        def rescale(self, units):
            return FakeQuantity(self._val, str(units))

    return SimpleNamespace(
        Quantity=lambda val, unit="": FakeQuantity(val, unit),
        dimensionless=FakeQuantity(1.0, "dimensionless"),
    )


class TestToQuantityRequiresPackage:
    def test_raises_import_error_without_quantities(self):
        with patch.dict(sys.modules, {"quantities": None}):
            from gwexpy.interop.quantities_ import to_quantity
            with pytest.raises(ImportError):
                to_quantity(TimeSeries(np.ones(5), t0=0, dt=1.0))


class TestToQuantity:
    def test_basic_conversion(self):
        pq = _fake_pq()
        with patch.dict(sys.modules, {"quantities": pq}):
            from gwexpy.interop import quantities_
            import importlib
            importlib.reload(quantities_)
            ts = TimeSeries(np.arange(5.0), t0=0, dt=1.0, unit="m")
            q = quantities_.to_quantity(ts)
        np.testing.assert_array_equal(q.magnitude, ts.value)

    def test_value_from_magnitude_attr(self):
        pq = _fake_pq()
        obj = SimpleNamespace(magnitude=np.array([1.0, 2.0]), unit=None)
        with patch.dict(sys.modules, {"quantities": pq}):
            from gwexpy.interop import quantities_
            import importlib
            importlib.reload(quantities_)
            q = quantities_.to_quantity(obj)
        np.testing.assert_array_equal(q.magnitude, np.array([1.0, 2.0]))

    def test_no_value_attr_uses_series_directly(self):
        pq = _fake_pq()
        data = np.array([3.0, 4.0])
        with patch.dict(sys.modules, {"quantities": pq}):
            from gwexpy.interop import quantities_
            import importlib
            importlib.reload(quantities_)
            q = quantities_.to_quantity(data)
        np.testing.assert_array_equal(q.magnitude, data)

    def test_none_unit_defaults_to_dimensionless(self):
        pq = _fake_pq()
        obj = SimpleNamespace(value=np.ones(3), unit=None)
        with patch.dict(sys.modules, {"quantities": pq}):
            from gwexpy.interop import quantities_
            import importlib
            importlib.reload(quantities_)
            q = quantities_.to_quantity(obj)
        assert q is not None  # Should not raise

    def test_unit_string_override(self):
        pq = _fake_pq()
        ts = TimeSeries(np.ones(3), t0=0, dt=1.0, unit="m")
        with patch.dict(sys.modules, {"quantities": pq}):
            from gwexpy.interop import quantities_
            import importlib
            importlib.reload(quantities_)
            q = quantities_.to_quantity(ts, units="km")
        assert q.units == "km"

    def test_lookup_error_with_dimensionless_fallback(self):
        class BadPQ:
            class Quantity:
                def __init__(self, val, unit_str=""):
                    if unit_str == "m":
                        raise LookupError("unknown unit")
                    self._val = np.asarray(val)
                    self._unit = unit_str

                def __mul__(self, other):
                    if isinstance(other, BadPQ.Quantity):
                        return BadPQ.Quantity.__new__(BadPQ.Quantity)
                    return BadPQ.Quantity(np.asarray(other) * 1.0, "")

                __rmul__ = __mul__

                @property
                def magnitude(self):
                    return self._val

            dimensionless = Quantity(1.0, "dimensionless")

        obj = SimpleNamespace(value=np.ones(3), unit=None)
        # unit=None -> u_str = "dimensionless" -> fallback triggers
        bad_pq = BadPQ()
        with patch.dict(sys.modules, {"quantities": bad_pq}):
            from gwexpy.interop import quantities_
            import importlib
            importlib.reload(quantities_)
            q = quantities_.to_quantity(obj)
        assert q is not None


class TestFromQuantity:
    def test_basic(self):
        from gwexpy.interop.quantities_ import from_quantity
        q = SimpleNamespace(
            magnitude=np.arange(5.0),
            dimensionality=SimpleNamespace(string="m"),
        )
        ts = from_quantity(TimeSeries, q, t0=0.0, dt=1.0)
        np.testing.assert_array_equal(ts.value, np.arange(5.0))
        assert str(ts.unit) == "m"

    def test_dimensionless_unit_converted_to_empty(self):
        from gwexpy.interop.quantities_ import from_quantity
        q = SimpleNamespace(
            magnitude=np.ones(3),
            dimensionality=SimpleNamespace(string="dimensionless"),
        )
        ts = from_quantity(TimeSeries, q, t0=0.0, dt=1.0)
        assert ts is not None

    def test_unit_kwarg_override(self):
        from gwexpy.interop.quantities_ import from_quantity
        q = SimpleNamespace(
            magnitude=np.ones(3),
            dimensionality=SimpleNamespace(string="m"),
        )
        ts = from_quantity(TimeSeries, q, t0=0.0, dt=1.0, unit="km")
        assert str(ts.unit) == "km"

    def test_t0_and_dt_passed_through(self):
        from gwexpy.interop.quantities_ import from_quantity
        q = SimpleNamespace(
            magnitude=np.ones(4),
            dimensionality=SimpleNamespace(string="s"),
        )
        ts = from_quantity(TimeSeries, q, t0=100.0, dt=0.5)
        assert ts.t0.value == pytest.approx(100.0)
        assert ts.dt.value == pytest.approx(0.5)
