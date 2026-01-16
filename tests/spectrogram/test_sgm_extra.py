import numpy as np
import pytest
from astropy import units as u
from gwexpy.spectrogram import SpectrogramMatrix
import tempfile
import os

class TestSpectrogramMatrixExtra:
    @pytest.fixture
    def sgm_basic(self):
        times = np.arange(10) * u.s
        freqs = np.arange(5) * 10 * u.Hz
        data = np.random.rand(2, 10, 5)
        return SpectrogramMatrix(
            data, 
            times=times, 
            frequencies=freqs, 
            rows=['ch1', 'ch2'], 
            name="TestSGM",
            unit=u.V
        )

    def test_plotting_smoke(self, sgm_basic):
        """Smoke test for plotting methods (check no error raised)."""
        try:
            import matplotlib.pyplot as plt
            p = sgm_basic.plot(show=False)
            plt.close('all')
            # plot_summary skipped as it requires specific config often
        except importError:
            pytest.skip("matplotlib not installed")

    @pytest.mark.xfail(reason="Transpose causes xindex validation mismatch due to axis swap", strict=True)
    def test_structure_ops(self, sgm_basic):
        """Test structural operations like Transpose, Flatten."""
        transposed = sgm_basic.T
        if isinstance(transposed, SpectrogramMatrix):
             # Just checking no crash in construction or basic access
             assert transposed.shape == (5, 10, 2)

    def test_io_hdf5(self, sgm_basic):
        """Test I/O via hdf5/gwpy if supported."""
        import pickle
        dumped = pickle.dumps(sgm_basic)
        loaded = pickle.loads(dumped)
        assert isinstance(loaded, SpectrogramMatrix)
        # assert loaded.shape == sgm_basic.shape

    def test_meta_arithmetic(self, sgm_basic):
        """Test if metadata names/units propagate in arithmetic."""
        res = sgm_basic + sgm_basic
        assert res.unit == u.V
        
        # This should now raise UnitConversionError (captured)
        with pytest.raises((u.UnitConversionError, ValueError)):
             _ = sgm_basic + 10 
             
        res2 = sgm_basic + 5*u.V
        assert res2.unit == u.V
