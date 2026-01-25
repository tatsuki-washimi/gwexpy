import numpy as np
import pytest

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
        assert fs.df.value == pytest.approx(1.0 / 12.0)
        assert len(fs) == 12 // 2 + 1  # 7

        # Check value (DC component sum matches)
        # padded sum = 10
        # dft[0] = sum / nfft = 10 / 12
        assert fs.value[0].real == pytest.approx(10.0 / 12.0)

    def test_fft_next_fast_len(self):
        # Prime number length to force larger next fast len
        data = np.zeros(101)
        ts = TimeSeries(data, dt=1.0)

        # mode transient
        fs = ts.fft(mode="transient", nfft_mode="next_fast_len")
        # next fast len for 101 is usually 108 or similar
        # Just check it's >= 101
        nfft = (len(fs) - 1) * 2
        if fs.frequencies[-1].value * 2 < (1.0 / ts.dt.value) * (nfft - 1) / nfft:
            # odd length case nfft = len(fs)*2 - 1 ? rfftfreq logic
            # actually we can just check fs.df
            nfft_calc = 1.0 / (fs.df.value * ts.dt.value)
            nfft = round(nfft_calc)

        assert nfft >= 101

    def test_transfer_function_steady_gwpy_match(self):
        """Verify steady mode matches GWpy transfer_function exactly."""
        gwpy_ts = pytest.importorskip("gwpy.timeseries")
        GwpyTimeSeries = gwpy_ts.TimeSeries

        np.random.seed(42)
        data1 = np.random.randn(1024)
        data2 = np.random.randn(1024)

        gwex_ts1 = TimeSeries(data1, sample_rate=64.0)
        gwex_ts2 = TimeSeries(data2, sample_rate=64.0)
        gwpy_ts1 = GwpyTimeSeries(data1, sample_rate=64.0)
        gwpy_ts2 = GwpyTimeSeries(data2, sample_rate=64.0)

        tf_gwex = gwex_ts1.transfer_function(
            gwex_ts2, mode="steady", fftlength=1.0, overlap=0
        )
        tf_gwpy = gwpy_ts1.transfer_function(gwpy_ts2, fftlength=1.0, overlap=0)

        np.testing.assert_allclose(
            tf_gwex.frequencies.value, tf_gwpy.frequencies.value, rtol=1e-12
        )
        np.testing.assert_allclose(tf_gwex.value, tf_gwpy.value, rtol=1e-10)

    def test_transfer_function_steady_zero_series_nan(self):
        data = np.zeros(1024)
        ts1 = TimeSeries(data, sample_rate=64.0)
        ts2 = TimeSeries(data, sample_rate=64.0)

        tf = ts1.transfer_function(ts2, mode="steady", fftlength=1.0, overlap=0)
        assert np.iscomplexobj(tf.value)
        assert np.isnan(tf.value.real).all()
        assert np.isnan(tf.value.imag).all()

    def test_transfer_function_transient_known_gain(self):
        """Verify transient mode with known constant gain using impulse input."""
        # Impulse input
        n_samples = 1000
        data_in = np.zeros(n_samples)
        data_in[0] = 1.0

        gain = 2.5
        data_out = gain * data_in

        ts_in = TimeSeries(data_in, sample_rate=1000.0)
        ts_out = TimeSeries(data_out, sample_rate=1000.0)

        tf = ts_in.transfer_function(ts_out, mode="transient")

        # All frequency bins should have the constant gain
        np.testing.assert_allclose(np.abs(tf.value), gain, rtol=1e-10)

    def test_transfer_function_transient_zero_zero_nan(self):
        n_samples = 128
        data_in = np.zeros(n_samples)
        data_out = np.zeros(n_samples)

        ts_in = TimeSeries(data_in, sample_rate=64.0)
        ts_out = TimeSeries(data_out, sample_rate=64.0)

        tf = ts_in.transfer_function(ts_out, mode="transient")
        assert np.iscomplexobj(tf.value)
        assert np.isnan(tf.value.real).all()
        assert np.isnan(tf.value.imag).all()

    def test_transfer_function_transient_signed_inf(self):
        n_samples = 128
        data_in = np.zeros(n_samples)

        ts_in = TimeSeries(data_in, sample_rate=64.0)

        data_out_pos = np.ones(n_samples)
        ts_out_pos = TimeSeries(data_out_pos, sample_rate=64.0)
        tf_pos = ts_in.transfer_function(ts_out_pos, mode="transient")
        assert np.isposinf(tf_pos.value[0].real)

        data_out_neg = -np.ones(n_samples)
        ts_out_neg = TimeSeries(data_out_neg, sample_rate=64.0)
        tf_neg = ts_in.transfer_function(ts_out_neg, mode="transient")
        assert np.isneginf(tf_neg.value[0].real)

    def test_transfer_function_transient_complex_phase_inf(self):
        n_samples = 128
        data_in = np.zeros(n_samples, dtype=complex)
        data_out = np.full(n_samples, 1.0 + 1.0j, dtype=complex)

        try:
            ts_in = TimeSeries(data_in, sample_rate=64.0)
            ts_out = TimeSeries(data_out, sample_rate=64.0)
            tf = ts_in.transfer_function(ts_out, mode="transient")
        except (TypeError, ValueError) as exc:
            pytest.skip(f"Complex transient FFT not supported: {exc}")

        fy = ts_out.fft(mode="transient")
        if np.imag(fy.value[0]) == 0:
            pytest.skip("Complex transient FFT does not preserve phase.")
        expected = (np.inf + 0j) * np.exp(1j * np.angle(fy.value[0]))
        np.testing.assert_allclose(tf.value[0], expected)

    def test_transfer_function_backward_compat(self):
        """Verify deprecated 'method' parameter still works with warning."""
        import warnings

        np.random.seed(42)
        data1 = np.random.randn(100)
        data2 = np.random.randn(100)
        ts1 = TimeSeries(data1, dt=1.0)
        ts2 = TimeSeries(data2, dt=1.0)

        # method="auto" with fftlength=None should use transient mode
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tf_fft = ts1.transfer_function(ts2, method="auto")
            assert any("deprecated" in str(warning.message).lower() for warning in w)

        # Verify it matches manual transient fft ratio
        f1 = ts1.fft(mode="transient")
        f2 = ts2.fft(mode="transient")
        tf_manual = f2 / f1
        np.testing.assert_allclose(tf_fft.value, tf_manual.value)

        # method="auto" with fftlength should use steady mode
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tf_welch = ts1.transfer_function(
                ts2, method="auto", fftlength=50, overlap=0, window="boxcar"
            )
            assert len(tf_welch) == 50 // 2 + 1

    def test_transfer_function_mismatch(self):
        # Create TS with different rates
        # Use enough samples to avoid filtering errors in resample
        ts1 = TimeSeries(np.zeros(1000), sample_rate=100)
        ts2 = TimeSeries(np.zeros(1000), sample_rate=50)

        # Error on mismatch
        with pytest.raises(ValueError, match="Sample rates differ"):
            ts1.transfer_function(ts2, mode="transient", downsample=False)

        # Warning and downsample
        with pytest.warns(UserWarning, match="Sample rates differ"):
            tf = ts1.transfer_function(ts2, mode="transient", downsample=None)
            # Alignment intersection.
            # ts1 (100Hz, 10s) -> [0, 10s)
            # ts2 (50Hz, 20s) -> [0, 20s)
            # Intersection [0, 10s).
            # Resample ts1 to 50Hz -> 500 samples.
            # Crop ts2 to [0, 10s) -> 500 samples.
            # FFT size 500.
            assert len(tf) == 251

        # No overlap with align="intersection"
        ts3 = TimeSeries(np.zeros(10), dt=1.0, t0=0)
        ts4 = TimeSeries(np.zeros(10), dt=1.0, t0=100)
        with pytest.raises(ValueError, match="No comparison overlap"):
            ts3.transfer_function(ts4, mode="transient", align="intersection")

        # Invalid align
        with pytest.raises(ValueError, match="align must be"):
            ts1.transfer_function(ts2, mode="transient", align="invalid")

    def test_fft_other_length(self):
        ts = TimeSeries(np.ones(10), dt=1.0)
        # other_length=10 -> target= 10 + 10 - 1 = 19
        fs = ts.fft(mode="transient", other_length=10)
        # nfft=19.
        # rfftfreq(19) -> 10 points
        assert len(fs) == 10
        assert fs.df.value == pytest.approx(1.0 / 19.0)
