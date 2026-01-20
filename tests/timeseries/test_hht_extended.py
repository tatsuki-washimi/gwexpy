
import numpy as np
import pytest
from astropy import units as u
from gwpy.spectrogram import Spectrogram

from gwexpy.timeseries import TimeSeries, TimeSeriesDict

PYEMD_AVAILABLE = False
try:
    import PyEMD
    # Basic check for EMD existence
    _ = PyEMD.EMD()
    PYEMD_AVAILABLE = True
except (ImportError, AttributeError):
    pass

class TestHHTExtended:
    @pytest.fixture
    def chirp_signal(self):
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        f0, f1 = 10, 50
        # Linear chirp
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2)
        data = np.sin(phase)
        return TimeSeries(data, dt=dt, unit="strain", name="chirp")

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_emd_residual_and_count(self, chirp_signal):
        # EMD
        res_emd = chirp_signal.emd(method="emd", return_residual=True)
        assert "residual" in res_emd
        # Count IMFs
        imf_keys = [k for k in res_emd.keys() if k.startswith("IMF")]
        assert len(imf_keys) > 0

        # EEMD
        res_eemd = chirp_signal.emd(method="eemd", eemd_trials=10, return_residual=True)
        assert "residual" in res_eemd
        imf_keys_eemd = [k for k in res_eemd.keys() if k.startswith("IMF")]
        assert len(imf_keys_eemd) > 0

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_emd_random_state(self, chirp_signal):
        # EEMD is stochastic
        res1 = chirp_signal.emd(method="eemd", eemd_trials=5, random_state=42)
        res2 = chirp_signal.emd(method="eemd", eemd_trials=5, random_state=42)
        res3 = chirp_signal.emd(method="eemd", eemd_trials=5, random_state=43)

        assert np.allclose(res1["IMF1"].value, res2["IMF1"].value)
        assert not np.allclose(res1["IMF1"].value, res3["IMF1"].value)

    def test_hilbert_analysis_extended(self, chirp_signal):
        # Test pad
        res_no_pad = chirp_signal.hilbert_analysis(pad=0)
        res_pad = chirp_signal.hilbert_analysis(pad=100)

        # Padding should change values near edges
        assert not np.allclose(res_no_pad["frequency"].value[:10], res_pad["frequency"].value[:10])
        # Use more relaxed tolerance for middle check as padding can have small global effects
        assert np.allclose(res_no_pad["frequency"].value[400:600], res_pad["frequency"].value[400:600], atol=0.5)

        # Test if_smooth
        res_smooth = chirp_signal.hilbert_analysis(if_smooth=11)
        # Smooth should have lower variance in small windows
        diff_orig = np.diff(res_no_pad["frequency"].value[400:500])
        diff_smooth = np.diff(res_smooth["frequency"].value[400:500])
        assert np.std(diff_smooth) < np.std(diff_orig)

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_hht_spectrogram_options(self, chirp_signal):
        # Test n_bins
        spec100 = chirp_signal.hht(output="spectrogram", n_bins=100, emd_method="emd")
        spec50 = chirp_signal.hht(output="spectrogram", n_bins=50, emd_method="emd")
        assert spec100.shape[1] == 100
        assert spec50.shape[1] == 50

        # Test fmin/fmax
        spec_range = chirp_signal.hht(output="spectrogram", fmin=20, fmax=40, n_bins=10, emd_method="emd")
        # Check that frequencies are within range plus/minus half bin width
        df = spec_range.frequencies[1].value - spec_range.frequencies[0].value
        assert spec_range.frequencies[0].value >= 20 - df
        assert spec_range.frequencies[-1].value <= 40 + df

        # Test weight='ia'
        spec_ia2 = chirp_signal.hht(output="spectrogram", weight="ia2", emd_method="emd")
        spec_ia = chirp_signal.hht(output="spectrogram", weight="ia", emd_method="emd")
        assert spec_ia2.unit == chirp_signal.unit ** 2
        assert spec_ia.unit == chirp_signal.unit
        # For a single chirp, ia2 should be approx ia^2. Use loose tolerance due to possible binning/residue effects.
        assert np.allclose(spec_ia2.value, spec_ia.value ** 2, atol=1e-1)


    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_hht_finite_only(self, chirp_signal):
        """
        Test finite_only by manually injecting NaN into IF/IA arrays.

        This approach does not rely on PyEMD's NaN tolerance and directly
        verifies that the spectrogram binning correctly excludes non-finite values.
        """
        from unittest.mock import patch

        # First do a normal HHT to get the shape
        spec_normal = chirp_signal.hht(output="spectrogram", emd_method="emd", finite_only=True)

        # Now test with patched hilbert_analysis that injects NaN
        original_hilbert_analysis = type(chirp_signal).hilbert_analysis
        call_count = [0]

        def patched_hilbert_analysis(self, **kwargs):
            result = original_hilbert_analysis(self, **kwargs)
            # Inject NaN into the middle of IF/IA
            call_count[0] += 1
            result["frequency"].value[500] = np.nan
            result["amplitude"].value[500] = np.nan
            return result

        with patch.object(type(chirp_signal), "hilbert_analysis", patched_hilbert_analysis):
            spec_with_nan = chirp_signal.hht(output="spectrogram", finite_only=True, emd_method="emd")

        # The spectrogram should have no NaN (finite_only excludes them)
        assert not np.any(np.isnan(spec_with_nan.value))

        # The sum should be slightly less due to dropped NaN points
        # (comparing at matching time index)
        assert spec_with_nan.value.sum() <= spec_normal.value.sum()

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_hht_if_policy_clip_vs_drop(self, chirp_signal):
        # Test if_policy='clip' includes more energy than 'drop' for restricted range
        spec_clip = chirp_signal.hht(output="spectrogram", fmin=20, fmax=30, if_policy="clip", emd_method="emd")
        spec_drop = chirp_signal.hht(output="spectrogram", fmin=20, fmax=30, if_policy="drop", emd_method="emd")
        assert spec_clip.value.sum() > spec_drop.value.sum()

    def test_hht_invalid_if_policy(self, chirp_signal):
        with pytest.raises(ValueError, match="Unknown if_policy"):
            chirp_signal.hht(output="spectrogram", if_policy="invalid", emd_method="emd")

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_hht_invalid_freq_bins(self, chirp_signal):
        # Non-monotonic freq_bins should raise
        bad_bins = np.array([10, 30, 20, 40])  # Not monotonic
        with pytest.raises(ValueError, match="monotonically increasing"):
            chirp_signal.hht(output="spectrogram", freq_bins=bad_bins, emd_method="emd")

    @pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")
    def test_hht_no_imfs_error(self):
        # Zero signal might result in no IMFs or just residual
        ts = TimeSeries(np.zeros(100), dt=0.01)
        with pytest.raises(ValueError, match="no IMFs"):
             ts.hht(output="spectrogram", emd_kwargs={"max_imf": 0})
