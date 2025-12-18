import pytest
import numpy as np
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.spectrogram import Spectrogram

PYEMD_AVAILABLE = False
try:
    import PyEMD
    # Basic check for EMD existence
    _ = PyEMD.EMD()
    PYEMD_AVAILABLE = True
except (ImportError, AttributeError):
    pass

class TestHHT:
    @pytest.fixture
    def sine_waves(self):
        # f1 = 10Hz, f2 = 5Hz combined with some trend
        dt = 0.01
        t = np.arange(0, 10.0, dt)
        
        # Non-stationary chirp: 5Hz to 10Hz
        # phi = 2*pi * (5*t + 0.5 * (5/10) * t^2)
        # f_inst = 5 + (5/10)*t = 5 + 0.5t
        phase = 2*np.pi * (5*t + 0.25 * t**2)
        chirp = np.sin(phase)
        
        # Simple Sine
        sine = np.sin(2 * np.pi * 10 * t)
        
        return TimeSeries(chirp, t0=0, dt=dt, unit="V", name="chirp"), \
               TimeSeries(sine, t0=0, dt=dt, unit="V", name="sine")

    def test_hilbert_analysis(self, sine_waves):
        chirp, sine = sine_waves
        # Just test hilbert analysis part (requires scipy only, which is likely present if gwpy works)
        # Assuming scipy is present
        
        res = sine.hilbert_analysis()
        assert "analytic" in res
        assert "amplitude" in res
        assert "phase" in res
        assert "frequency" in res
        
        # Sine wave frequency ~ 10Hz
        freq = res["frequency"]
        # Skip edges where gradient might be weird
        mid_freq = freq[10:-10].value
        assert np.allclose(mid_freq, 10.0, atol=1.0) # Within 1Hz tolerance
        
        # Amplitude should be ~ 1
        amp = res["amplitude"]
        assert np.allclose(amp[10:-10].value, 1.0, atol=0.1)

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_emd_simple(self, sine_waves):
        chirp, sine = sine_waves
        
        # Standard EMD
        imfs = sine.emd(method="emd", max_imf=2)
        assert isinstance(imfs, TimeSeriesDict)
        
        # Sine might be single mode, but maybe noise/precision introduces others.
        # Check keys
        assert "IMF1" in imfs
        
        # Sum of IMFs + residual should reconstruct signal approx
        recon = np.zeros_like(sine.value)
        for k in imfs:
             recon += imfs[k].value
             
        assert np.allclose(recon, sine.value, atol=1e-5)

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_hht_dict_output(self, sine_waves):
        chirp, sine = sine_waves
        
        res = chirp.hht(method="emd", output="dict")
        assert isinstance(res, dict)
        assert "imfs" in res
        assert "ia" in res
        assert "if" in res
        
        # Check if frequencies match chirp trend
        # f_inst = 5 + 0.5t
        # IMF1 likely captures the main chirp
        if_series = res["if"]["IMF1"]
        t = if_series.times.value
        expected_f = 5 + 0.5 * t
        
        # loose check
        # EMD boundaries are tricky
        diff = np.abs(if_series.value - expected_f)
        # Check median error is small
        assert np.median(diff[10:-10]) < 2.0 # 2Hz margin

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_hht_spectrogram(self, sine_waves):
        chirp, sine = sine_waves
        
        res = chirp.hht(output="spectrogram")
        assert isinstance(res, Spectrogram)
        assert res.shape[0] == len(chirp) # Time axis
        assert res.shape[1] == 100 # Default frequency bins used
        
        # Check energy exists
        assert res.value.max() > 0

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_tsd_hht(self, sine_waves):
        chirp, sine = sine_waves
        data = TimeSeriesDict()
        data["C1"] = chirp
        data["S1"] = sine
        
        res = data.hht(output="dict") # Pass output param to ts.hht
        assert isinstance(res, dict)
        assert "C1" in res
        assert "S1" in res
        
        assert "imfs" in res["C1"]
