import pytest
import numpy as np
from gwexpy.timeseries import TimeSeries

class TestTimeSeriesExtensions:
    def test_fft_default(self):
        # Create random TS
        np.random.seed(42)
        data = np.random.randn(100)
        ts = TimeSeries(data, dt=0.01)

        # Default fft (gwpy mode implicit)
        fs1 = ts.fft()
        fs2 = ts.fft(mode="gwpy")

        np.testing.assert_array_equal(fs1.value, fs2.value)
        assert fs1.size == fs2.size

        # Compare with explicit manual calculation matching GWpy rules
        # rule: rfft(x)/N, dft[1:]*=2
        dft = np.fft.rfft(data) / 100.0
        dft[1:] *= 2.0
        np.testing.assert_allclose(fs1.value, dft)

    def test_fft_transient_padding(self):
        data = np.ones(10)
        ts = TimeSeries(data, dt=1.0)

        # pad_left=1, pad_right=1 -> size 12
        fs = ts.fft(mode="transient", pad_left=1, pad_right=1, pad_mode="zero")

        # Frequencies
        # nfft=12. df = 1 / (12 * 1) = 1/12 Hz.
        assert fs.df.value == pytest.approx(1.0/12.0)
        assert len(fs) == 12 // 2 + 1 # 7

        # Check value (DC component sum matches)
        # padded sum = 10
        # dft[0] = sum / nfft = 10 / 12
        assert fs.value[0].real == pytest.approx(10.0/12.0)

    def test_fft_next_fast_len(self):
        # Prime number length to force larger next fast len
        data = np.zeros(101)
        ts = TimeSeries(data, dt=1.0)

        # mode transient
        fs = ts.fft(mode="transient", nfft_mode="next_fast_len")
        # next fast len for 101 is usually 108 or similar
        # Just check it's >= 101
        nfft = (len(fs) - 1) * 2
        if fs.frequencies[-1].value * 2 < (1.0/ts.dt.value) * (nfft-1)/nfft:
             # odd length case nfft = len(fs)*2 - 1 ? rfftfreq logic
             # actually we can just check fs.df
             nfft_calc = 1.0 / (fs.df.value * ts.dt.value)
             nfft = round(nfft_calc)

        assert nfft >= 101

    def test_transfer_function_auto(self):
        np.random.seed(42)
        data1 = np.random.randn(100)
        data2 = np.random.randn(100)
        ts1 = TimeSeries(data1, dt=1.0)
        ts2 = TimeSeries(data2, dt=1.0)

        # fftlength=None -> FFT ratio
        # method="auto" implies method="fft" when fftlength is None
        tf_fft = ts1.transfer_function(ts2, method="auto")

        # Verify it matches manual fft ratio
        # Note: manual ratio needs checking normalization.
        # fft() returns normalized by N.
        # ratio (Y/N) / (X/N) = Y/X. Scale factors cancel out.
        f1 = ts1.fft()
        f2 = ts2.fft()
        tf_manual = f2 / f1
        np.testing.assert_allclose(tf_fft.value, tf_manual.value)

        # fftlength given -> CSD/PSD (runs usage of welch)
        # Just check it runs and returns different result or specific size
        tf_welch = ts1.transfer_function(ts2, method="auto", fftlength=50, overlap=0, window='boxcar')
        assert len(tf_welch) == 50//2 + 1

    def test_transfer_function_mismatch(self):
        # Create TS with different rates
        # Use enough samples to avoid filtering errors in resample
        ts1 = TimeSeries(np.zeros(1000), sample_rate=100)
        ts2 = TimeSeries(np.zeros(1000), sample_rate=50)

        # Error on mismatch
        with pytest.raises(ValueError, match="Sample rates differ"):
            ts1.transfer_function(ts2, method="fft", downsample=False)

        # Warning and downsample
        with pytest.warns(UserWarning, match="Sample rates differ"):
            tf = ts1.transfer_function(ts2, method="fft", downsample=None)
            # Alignment intersection.
            # ts1 (100Hz, 10s) -> [0, 10s)
            # ts2 (50Hz, 20s) -> [0, 20s)
            # Intersection [0, 10s).
            # Resample ts1 to 50Hz -> 500 samples.
            # Crop ts2 to [0, 10s) -> 500 samples.
            # FFT size 500.
            assert len(tf) == 251

    def test_fft_other_length(self):
         ts = TimeSeries(np.ones(10), dt=1.0)
         # other_length=10 -> target= 10 + 10 - 1 = 19
         fs = ts.fft(mode="transient", other_length=10)
         # nfft=19.
         # rfftfreq(19) -> 10 points
         assert len(fs) == 10
         assert fs.df.value == pytest.approx(1.0/19.0)
