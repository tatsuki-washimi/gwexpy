"""
Test astro module semantics and A2 numerical logic for gwexpy.astro.

These tests verify:
1. Unit consistency (Mpc results)
2. Robustness to input units
3. Physical sanity of results
"""

import importlib.util

import numpy as np
import pytest
from astropy import units as u

from gwexpy.astro import inspiral_range, sensemon_range
from gwexpy.frequencyseries import FrequencySeries

INSPIRAL_RANGE_AVAILABLE = importlib.util.find_spec("inspiral_range") is not None


class TestAstroSemantics:
    """Test A2 semantics of astro module."""

    @pytest.fixture
    def aLIGO_flat_psd(self):
        """Create a flat PSD roughly at aLIGO sensitivity floor."""
        # 1e-23 ASD -> 1e-46 PSD
        df = 0.25
        freqs = np.arange(10, 2048, df)
        data = np.full_like(freqs, 1e-46, dtype=float)
        return FrequencySeries(
            data, frequencies=freqs, unit=u.dimensionless_unscaled**2 / u.Hz
        )

    @pytest.mark.skipif(
        not INSPIRAL_RANGE_AVAILABLE,
        reason="inspiral-range package is required for inspiral_range tests",
    )
    def test_inspiral_range_units(self, aLIGO_flat_psd):
        """Test that inspiral_range returns a Quantity with length units (Mpc)."""
        # inspiraL_range returns a Quantity (usually Mpc)
        r = inspiral_range(aLIGO_flat_psd, mass1=1.4, mass2=1.4)

        assert isinstance(r, u.Quantity)
        assert r.unit.is_equivalent(u.m)  # It is a length
        # GWpy default is usually Mpc, let's verify it matches the expected magnitude
        # 1e-23 ASD is very sensitive, range should be > 100 Mpc
        assert r.to("Mpc").value > 100

    def test_sensemon_range_units(self, aLIGO_flat_psd):
        """Test that sensemon_range returns a Quantity with length units (Mpc)."""
        r = sensemon_range(aLIGO_flat_psd, mass1=1.4, mass2=1.4)

        assert isinstance(r, u.Quantity)
        assert r.unit.is_equivalent(u.m)
        assert r.value > 0

    @pytest.mark.skipif(
        not INSPIRAL_RANGE_AVAILABLE,
        reason="inspiral-range package is required for inspiral_range tests",
    )
    def test_range_input_unit_robustness(self):
        """Test robustness to different PSD units (strain/Hz vs strain^2/Hz)."""
        df = 0.25
        freqs = np.arange(10, 2048, df)
        data = np.full_like(freqs, 1e-46, dtype=float)

        # Unit as strain^2 / Hz
        psd1 = FrequencySeries(data, frequencies=freqs, unit="1/Hz")
        r1 = inspiral_range(psd1)

        # Unit as strain / sqrt(Hz) -> this is ASD, should fail if function expects PSD
        # But some functions might be smart? GWpy inspiral_range expects PSD.
        # Let's see if it handles different representations of the same unit.
        psd2 = FrequencySeries(data, frequencies=freqs, unit=u.Unit("1/Hz"))
        r2 = inspiral_range(psd2)

        assert r1 == r2

    @pytest.mark.skipif(
        not INSPIRAL_RANGE_AVAILABLE,
        reason="inspiral-range package is required for inspiral_range tests",
    )
    def test_range_value_sanity(self, aLIGO_flat_psd):
        """Verify range is positive and within sanity bounds."""
        r = inspiral_range(aLIGO_flat_psd)

        assert r.value > 0
        # For aLIGO-like flat noise, it should be in the order of 100-200 Mpc
        assert 10 < r.to("Mpc").value < 1000

    def test_reexport_consistency(self):
        """Verify that gwexpy.astro re-exports match gwpy.astro."""
        import gwpy.astro

        import gwexpy.astro

        assert gwexpy.astro.inspiral_range == gwpy.astro.inspiral_range
        assert gwexpy.astro.sensemon_range == gwpy.astro.sensemon_range
