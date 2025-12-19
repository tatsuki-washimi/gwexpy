
import pytest
import numpy as np
from gwexpy.timeseries import TimeSeriesMatrix
import astropy.units as u

class TestTimeSeriesMatrixLockIn:
    def test_lock_in_amp_phase(self):
        # Create dummy matrix
        data = np.random.randn(2, 2, 100)
        times = np.arange(100) * 0.01
        tsm = TimeSeriesMatrix(data, times=times, unit='V')
        
        # Test 1: amp_phase (tuple return)
        res = tsm.lock_in(f0=10, output='amp_phase')
        assert isinstance(res, tuple)
        assert len(res) == 2
        amp, phase = res
        assert isinstance(amp, TimeSeriesMatrix)
        assert isinstance(phase, TimeSeriesMatrix)
        assert amp.shape == (2, 2, 1)
        assert phase.shape == (2, 2, 1)
        assert phase[0,0].unit == u.deg
        
    def test_lock_in_iq(self):
        data = np.random.randn(2, 1, 100)
        times = np.arange(100) * 0.01
        tsm = TimeSeriesMatrix(data, times=times, unit='V')
        
        # Test 2: iq (tuple return)
        i, q = tsm.lock_in(f0=10, output='iq')
        assert isinstance(i, TimeSeriesMatrix)
        assert isinstance(q, TimeSeriesMatrix)
        
    def test_lock_in_complex(self):
        data = np.random.randn(1, 2, 100)
        times = np.arange(100) * 0.01
        tsm = TimeSeriesMatrix(data, times=times, unit='V')

        # Test 3: complex (single return)
        c = tsm.lock_in(f0=10, output='complex')
        assert isinstance(c, TimeSeriesMatrix)
        assert np.iscomplexobj(c.value)

