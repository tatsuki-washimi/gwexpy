
import pytest
import numpy as np
from gwexpy.frequencyseries import FrequencySeries

class TestSpectralInterop:
    
    def test_specutils_interop(self):
        """Test FrequencySeries <-> specutils.Spectrum1D conversion."""
        pytest.importorskip("specutils")
        from specutils import Spectrum1D
        
        # 1. Create FrequencySeries
        data = np.random.randn(100)
        frequencies = np.linspace(0, 100, 100)
        fs = FrequencySeries(data, frequencies=frequencies, unit='V', name='test_spec')
        
        # 2. To Spectrum1D
        spec = fs.to_specutils()
        
        assert isinstance(spec, Spectrum1D)
        assert len(spec.flux) == 100
        # Check units if possible
        # spec.flux should be Quantity(value=data, unit=V) ideally?
        # But our implementation multiplied by unit.
        
        # Check values
        np.testing.assert_allclose(spec.flux.value, data)
        np.testing.assert_allclose(spec.spectral_axis.value, frequencies)
        
        # 3. From Spectrum1D
        fs_rec = FrequencySeries.from_specutils(spec)
        
        assert isinstance(fs_rec, FrequencySeries)
        np.testing.assert_allclose(fs_rec.value, data) # fs_rec.value is numpy array
        np.testing.assert_allclose(fs_rec.frequencies.value, frequencies)
        
    def test_pyspeckit_interop(self):
        """Test FrequencySeries <-> pyspeckit.Spectrum conversion."""
        pytest.importorskip("pyspeckit")
        import pyspeckit
        
        # 1. Create FrequencySeries
        data = np.random.randn(50)
        frequencies = np.linspace(0, 50, 50)
        fs = FrequencySeries(data, frequencies=frequencies, unit='V', name='test_pyspeckit')
        
        # 2. To pyspeckit
        sp = fs.to_pyspeckit()
        
        assert isinstance(sp, pyspeckit.Spectrum)
        assert sp.data.shape == (50,)
        
        # Check values
        np.testing.assert_allclose(sp.data, data)
        np.testing.assert_allclose(sp.xarr, frequencies)
        
        # 3. From pyspeckit
        fs_rec = FrequencySeries.from_pyspeckit(sp)
        
        assert isinstance(fs_rec, FrequencySeries)
        np.testing.assert_allclose(fs_rec.value, data)
        np.testing.assert_allclose(fs_rec.frequencies.value, frequencies)
