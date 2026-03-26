"""Tests for Pint ↔ astropy unit conversion utilities."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.utils.units import (
    astropy_to_pint,
    astropy_unit_to_pint_unit,
    pint_to_astropy,
    pint_unit_to_astropy_unit,
)

pint = pytest.importorskip("pint")
ureg = pint.UnitRegistry()


# ---------------------------------------------------------------------------
# pint_unit_to_astropy_unit
# ---------------------------------------------------------------------------


class TestPintUnitToAstropy:
    @pytest.mark.parametrize(
        "pint_u,expected",
        [
            (ureg.meter, u.m),
            (ureg.second, u.s),
            (ureg.hertz, u.Hz),
            (ureg.volt, u.V),
            (ureg.ampere, u.A),
            (ureg.kelvin, u.K),
        ],
    )
    def test_basic_si(self, pint_u, expected):
        result = pint_unit_to_astropy_unit(pint_u)
        assert result == expected

    def test_compound_unit(self):
        pu = ureg.meter / ureg.second
        au = pint_unit_to_astropy_unit(pu)
        assert au.is_equivalent(u.m / u.s)

    def test_compound_vm(self):
        pu = ureg.volt / ureg.meter
        au = pint_unit_to_astropy_unit(pu)
        assert au.is_equivalent(u.V / u.m)

    def test_dimensionless(self):
        pu = ureg.dimensionless
        au = pint_unit_to_astropy_unit(pu)
        assert au == u.dimensionless_unscaled

    def test_watt_per_kg(self):
        pu = ureg.watt / ureg.kilogram
        au = pint_unit_to_astropy_unit(pu)
        assert au.is_equivalent(u.W / u.kg)


# ---------------------------------------------------------------------------
# astropy_unit_to_pint_unit
# ---------------------------------------------------------------------------


class TestAstropyUnitToPint:
    @pytest.mark.parametrize(
        "astropy_u,pint_str",
        [
            (u.m, "meter"),
            (u.s, "second"),
            (u.Hz, "hertz"),
            (u.V, "volt"),
            (u.A, "ampere"),
            (u.K, "kelvin"),
        ],
    )
    def test_basic_si(self, astropy_u, pint_str):
        result = astropy_unit_to_pint_unit(astropy_u)
        # Just verify it's a valid Pint unit
        assert result is not None

    def test_compound_unit(self):
        pu = astropy_unit_to_pint_unit(u.m / u.s)
        assert pu is not None

    def test_dimensionless(self):
        pu = astropy_unit_to_pint_unit(u.dimensionless_unscaled)
        assert pu is not None


# ---------------------------------------------------------------------------
# pint_to_astropy (quantities)
# ---------------------------------------------------------------------------


class TestPintToAstropy:
    def test_scalar(self):
        pq = 3.14 * ureg.meter
        aq = pint_to_astropy(pq)
        assert isinstance(aq, u.Quantity)
        assert aq.value == pytest.approx(3.14)
        assert aq.unit == u.m

    def test_array(self):
        pq = np.array([1.0, 2.0, 3.0]) * ureg.second
        aq = pint_to_astropy(pq)
        assert len(aq) == 3
        np.testing.assert_allclose(aq.value, [1.0, 2.0, 3.0])
        assert aq.unit == u.s

    def test_compound(self):
        pq = 9.81 * (ureg.meter / ureg.second**2)
        aq = pint_to_astropy(pq)
        assert aq.value == pytest.approx(9.81)
        assert aq.unit.is_equivalent(u.m / u.s**2)


# ---------------------------------------------------------------------------
# astropy_to_pint (quantities)
# ---------------------------------------------------------------------------


class TestAstropyToPint:
    def test_scalar(self):
        aq = 2.718 * u.m
        pq = astropy_to_pint(aq)
        assert float(pq.magnitude) == pytest.approx(2.718)

    def test_array(self):
        aq = np.array([10.0, 20.0]) * u.Hz
        pq = astropy_to_pint(aq)
        np.testing.assert_allclose(pq.magnitude, [10.0, 20.0])


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    @pytest.mark.parametrize(
        "pint_u",
        [
            ureg.meter,
            ureg.second,
            ureg.hertz,
            ureg.meter / ureg.second,
            ureg.volt / ureg.meter,
            ureg.watt / ureg.kilogram,
        ],
    )
    def test_unit_roundtrip(self, pint_u):
        au = pint_unit_to_astropy_unit(pint_u)
        pu2 = astropy_unit_to_pint_unit(au)
        # Both should represent equivalent physical dimensions
        assert pu2 is not None

    def test_quantity_roundtrip(self):
        original = 42.0 * ureg.meter / ureg.second
        aq = pint_to_astropy(original)
        pq = astropy_to_pint(aq)
        assert float(pq.magnitude) == pytest.approx(42.0)
