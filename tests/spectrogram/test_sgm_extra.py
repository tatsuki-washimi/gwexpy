"""
Extra tests for SpectrogramMatrix features.

These tests cover plotting, structural operations, I/O, and metadata arithmetic.
"""

import os
import pickle
import tempfile

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram, SpectrogramMatrix
from gwexpy.spectrogram.collections import SpectrogramList


class TestSpectrogramMatrixExtra:
    """Extra tests for SpectrogramMatrix functionality."""

    @pytest.fixture
    def times(self):
        return np.arange(10) * u.s

    @pytest.fixture
    def frequencies(self):
        return np.arange(5) * 10 * u.Hz

    @pytest.fixture
    def sgm_basic(self, times, frequencies):
        """Basic SpectrogramMatrix via SpectrogramList.to_matrix()."""
        # Use SpectrogramList.to_matrix() to avoid xindex vs x0 warnings
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=frequencies,
            unit=u.V,
            name="ch1",
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=frequencies,
            unit=u.V,
            name="ch2",
        )
        matrix = SpectrogramList([sg1, sg2]).to_matrix()
        matrix.name = "TestSGM"
        return matrix

    def test_plotting_smoke(self, sgm_basic):
        """Smoke test for plotting methods (check no error raised)."""
        try:
            import matplotlib.pyplot as plt

            _ = sgm_basic.plot(show=False)
            plt.close("all")
        except ImportError:
            pytest.skip("matplotlib not installed")

    @pytest.mark.xfail(
        reason="Transpose swaps rows/cols but xindex (time) length validation fails "
        "because SpectrogramMatrix expects xindex.length == shape[-2] (time axis), "
        "but after T the shape becomes (cols, rows, time) breaking this invariant. "
        "Fixing requires 4D support or axis-swap-aware xindex handling.",
        strict=True,
    )
    def test_structure_ops(self, sgm_basic):
        """Test structural operations like Transpose."""
        transposed = sgm_basic.T
        if isinstance(transposed, SpectrogramMatrix):
            # After transpose (1, 0, 2), shape (2, 10, 5) -> (10, 2, 5)
            # But this is semantically wrong for SpectrogramMatrix
            assert transposed.shape[0] != sgm_basic.shape[0]

    def test_pickle_round_trip(self, sgm_basic):
        """Test pickle serialization/deserialization.

        Current limitation: pickle round-trip preserves shape and values,
        but xindex/frequencies may be lost due to numpy ndarray subclass
        pickling behavior. This test documents current behavior.
        """
        # Serialize and deserialize
        dumped = pickle.dumps(sgm_basic)
        loaded = pickle.loads(dumped)

        # Type check
        assert isinstance(loaded, SpectrogramMatrix)

        # Shape preserved
        assert loaded.shape == sgm_basic.shape

        # Values preserved (exact match)
        assert np.array_equal(loaded.view(np.ndarray), sgm_basic.view(np.ndarray))

        # Note: xindex/frequencies are not preserved in pickle round-trip.
        # This is a known limitation of numpy ndarray subclass pickling.
        # When __reduce_ex__ is not overridden, xindex becomes None after unpickling.
        # For now, we only assert that the loaded object is valid.
        # Future: implement __reduce_ex__ or __getstate__/__setstate__ for full support.

    def test_meta_arithmetic(self, sgm_basic):
        """Test if metadata names/units propagate in arithmetic."""
        res = sgm_basic + sgm_basic
        # New design: units in meta, not global. Check per-element unit.
        assert res.meta[0, 0].unit == u.V
        assert res.meta[1, 0].unit == u.V

        res2 = sgm_basic + 5 * u.V
        assert res2.meta[0, 0].unit == u.V
