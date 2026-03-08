"""Tests for gwexpy.numerics.constants."""

import numpy as np
import pytest

from gwexpy.numerics.constants import (
    EPS_COHERENCE,
    EPS_PSD,
    EPS_VARIANCE,
    SAFE_FLOOR,
    SAFE_FLOOR_STRAIN,
    eps_for_dtype,
)


class TestEpsForDtype:
    def test_float64(self):
        eps = eps_for_dtype(np.float64)
        assert eps == pytest.approx(np.finfo(np.float64).eps)

    def test_float32(self):
        eps = eps_for_dtype(np.float32)
        assert eps == pytest.approx(np.finfo(np.float32).eps)

    def test_default_is_float64(self):
        assert eps_for_dtype() == eps_for_dtype(np.float64)

    def test_string_dtype(self):
        assert eps_for_dtype("float32") == eps_for_dtype(np.float32)

    def test_returns_float(self):
        assert isinstance(eps_for_dtype(), float)


class TestSemanticConstants:
    def test_eps_variance_positive(self):
        assert EPS_VARIANCE > 0

    def test_eps_psd_positive(self):
        assert EPS_PSD > 0

    def test_eps_coherence_positive(self):
        assert EPS_COHERENCE > 0

    def test_ordering(self):
        """EPS_COHERENCE < EPS_VARIANCE < EPS_PSD by design."""
        assert EPS_COHERENCE < EPS_VARIANCE < EPS_PSD

    def test_safe_floor(self):
        assert SAFE_FLOOR == 1e-50

    def test_safe_floor_strain_alias(self):
        assert SAFE_FLOOR_STRAIN == SAFE_FLOOR

    def test_safe_floor_below_gw_power(self):
        """SAFE_FLOOR should be well below GW strain power (~1e-42)."""
        assert SAFE_FLOOR < 1e-42

    def test_log_safe(self):
        """SAFE_FLOOR should prevent log(0)."""
        result = np.log10(0.0 + SAFE_FLOOR)
        assert np.isfinite(result)
        assert result == pytest.approx(-50.0)
