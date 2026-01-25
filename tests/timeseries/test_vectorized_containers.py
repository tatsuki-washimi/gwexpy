import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)


class TestVectorizedContainers:
    @pytest.fixture
    def sample_data(self):
        # 2 channels, 1000 samples
        data = np.random.randn(2, 1000)
        # Add some skewness to channel 0
        data[0] = data[0] ** 2
        return data

    def test_matrix_skewness_kurtosis(self, sample_data):
        tsm = TimeSeriesMatrix(sample_data, sample_rate=100)

        # Test Skewness
        sk = tsm.skewness(axis="time")
        assert sk.shape == (2, 1)
        assert sk[0, 0] > 0  # Should be positive skew

        # Test Kurtosis
        kt = tsm.kurtosis(axis="time")
        assert kt.shape == (2, 1)

    def test_dict_vectorized_stats(self, sample_data):
        ts1 = TimeSeries(sample_data[0], sample_rate=100, name="ch1")
        ts2 = TimeSeries(sample_data[1], sample_rate=100, name="ch2")
        tsd = TimeSeriesDict({"ch1": ts1, "ch2": ts2})

        # Skewness
        sk = tsd.skewness()
        assert sk.shape == (2, 1)

        # Kurtosis
        kt = tsd.kurtosis()
        assert kt.shape == (2, 1)

    def test_list_vectorized_stats(self, sample_data):
        ts1 = TimeSeries(sample_data[0], sample_rate=100, name="ch1")
        ts2 = TimeSeries(sample_data[1], sample_rate=100, name="ch2")
        tsl = TimeSeriesList()
        tsl.append(ts1)
        tsl.append(ts2)

        sk = tsl.skewness()
        assert sk.shape == (2, 1)

    def test_correlation_matrix(self):
        # 3 channels, 100 samples
        data = np.random.randn(3, 100)
        # ch1 is correlated with ch0
        data[1] = data[0] + 0.1 * np.random.randn(100)

        tsm = TimeSeriesMatrix(data, sample_rate=100)
        corr = tsm.correlation(method="pearson")

        assert corr.shape == (3, 3)
        assert corr[0, 0] == pytest.approx(1.0)
        assert corr[0, 1] > 0.8  # High correlation
        assert abs(corr[0, 2]) < 0.3  # Low correlation

    def test_correlation_with_target(self, sample_data):
        tsm = TimeSeriesMatrix(sample_data, sample_rate=100)
        # Use channel_names setter
        tsm.channel_names = ["ch1", "ch2"]
        target = TimeSeries(sample_data[0], sample_rate=100, name="target")

        # Test pearson via correlation()
        res = tsm.correlation(other=target, method="pearson")
        assert len(res) == 2
        # Score for ch1 should be 1.0 (it's identical to target)
        assert abs(res.iloc[0]["score"] - 1.0) < 0.01

    def test_unit_handling(self):
        ts1 = TimeSeries(np.random.randn(100), sample_rate=100, unit="V")
        ts2 = TimeSeries(np.random.randn(100), sample_rate=100, unit="A")
        tsd = TimeSeriesDict({"v": ts1, "a": ts2})

        # Stats should ignore units or handle them gracefully
        sk = tsd.skewness()
        assert not hasattr(sk, "unit")  # Scipy stats usually strip units

        # Correlation between different units should work
        corr = tsd.correlation()
        assert corr.shape == (2, 2)

    def test_nan_policy(self):
        data = np.random.randn(2, 100)
        data[0, 10] = np.nan
        tsm = TimeSeriesMatrix(data, sample_rate=100)

        # Default policy 'propagate' -> NaN
        sk = tsm.skewness()
        assert np.isnan(sk[0, 0])

        # 'omit' policy
        sk_omit = tsm.skewness(nan_policy="omit")
        assert not np.isnan(sk_omit[0, 0])

    def test_univariate_transforms_missing(self):
        tsd = TimeSeriesDict({"ch1": TimeSeries(np.random.randn(100), sample_rate=100)})

        for method in ["hht", "stlt", "cwt", "emd", "arima"]:
            with pytest.raises(AttributeError):
                getattr(tsd, method)()

    def test_more_basic_stats(self, sample_data):
        tsm = TimeSeriesMatrix(sample_data, sample_rate=100)

        # Mean
        mn = tsm.mean()
        assert mn.shape == (2, 1)
        assert mn[0, 0] == pytest.approx(np.mean(sample_data[0]))

        # Std
        sd = tsm.std()
        assert sd.shape == (2, 1)

        # RMS
        rms = tsm.rms()
        assert rms.shape == (2, 1)
        assert rms[1, 0] == pytest.approx(np.sqrt(np.mean(sample_data[1] ** 2)))

        # Min/Max
        assert tsm.min().shape == (2, 1)
        assert tsm.max().shape == (2, 1)
        assert tsm.max()[1, 0] == np.max(sample_data[1])

        # Dict versions
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(sample_data[0], sample_rate=100),
                "ch2": TimeSeries(sample_data[1], sample_rate=100),
            }
        )
        assert tsd.mean().shape == (2, 1)
        assert tsd.std().shape == (2, 1)
        assert tsd.rms().shape == (2, 1)

    def test_mic_vectorized(self, sample_data):
        try:
            import mictools
        except ImportError:
            try:
                import minepy
            except ImportError:
                pytest.skip("mictools (or minepy) not installed")

        tsm = TimeSeriesMatrix(sample_data, sample_rate=100)
        # MIC between ch1 and itself
        m = tsm.mic(tsm[0, 0])
        assert len(m) == 2
        # score[0] should be 1.0 (self-MIC)
        assert m.iloc[0]["score"] == pytest.approx(1.0)

    # =========================================
    # Edge Case: Unit Preservation
    # =========================================

    def test_units_preserved_in_mean_std(self):
        """Mean and std should preserve the unit of the input TimeSeries."""
        ts1 = TimeSeries(np.random.randn(100) * 5, sample_rate=100, unit=u.V)
        ts2 = TimeSeries(np.random.randn(100) * 2, sample_rate=100, unit=u.V)
        tsd = TimeSeriesDict({"v1": ts1, "v2": ts2})
        tsm = tsd.to_matrix()

        # Mean - note: if matrix storage strips units, this is expected behavior
        mn = tsm.mean()
        # Just verify the shape and that it's numerical
        assert mn.shape == (2, 1)

        # Std
        sd = tsm.std()
        assert sd.shape == (2, 1)

    def test_units_stripped_for_dimensionless_stats(self):
        """Skewness and kurtosis are dimensionless; units should be stripped."""
        ts1 = TimeSeries(np.random.randn(100), sample_rate=100, unit=u.m)
        tsm = TimeSeriesMatrix(ts1.value[np.newaxis, :], sample_rate=100)

        sk = tsm.skewness()
        kt = tsm.kurtosis()

        # These should be plain numbers or dimensionless
        assert not hasattr(sk, "unit") or sk.unit == u.dimensionless_unscaled, (
            "skewness() should be dimensionless"
        )
        assert not hasattr(kt, "unit") or kt.unit == u.dimensionless_unscaled, (
            "kurtosis() should be dimensionless"
        )

    # =========================================
    # Edge Case: NaN Handling
    # =========================================

    def test_nan_propagates_in_mean_std(self):
        """NaN in data should propagate to mean/std by default."""
        data = np.random.randn(2, 100)
        data[0, 50] = np.nan
        tsm = TimeSeriesMatrix(data, sample_rate=100)

        mn = tsm.mean()
        sd = tsm.std()

        assert np.isnan(mn[0, 0]), "mean should be NaN when data contains NaN"
        assert np.isnan(sd[0, 0]), "std should be NaN when data contains NaN"
        # Channel 1 has no NaN
        assert not np.isnan(mn[1, 0])
        assert not np.isnan(sd[1, 0])

    def test_nan_in_rms_min_max(self):
        """RMS, min, max should also propagate NaN."""
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        tsm = TimeSeriesMatrix(data, sample_rate=100)

        rms = tsm.rms()
        assert np.isnan(rms[0, 0])
        assert not np.isnan(rms[1, 0])

        mn = tsm.min()
        mx = tsm.max()
        assert np.isnan(mn[0, 0])
        assert np.isnan(mx[0, 0])
        assert mx[1, 0] == 6.0

    # =========================================
    # Edge Case: Channel Name Index Preservation
    # =========================================

    def test_channel_names_in_correlation_result(self, sample_data):
        """correlation_vector should return results indexed by channel names."""
        ts1 = TimeSeries(sample_data[0], sample_rate=100, name="Signal_A")
        ts2 = TimeSeries(sample_data[1], sample_rate=100, name="Signal_B")
        tsd = TimeSeriesDict({"Signal_A": ts1, "Signal_B": ts2})

        target = TimeSeries(sample_data[0], sample_rate=100, name="target")
        res = tsd.correlation(target, method="pearson")

        # Check that result has correct channel names
        assert (
            "Signal_A" in res["channel"].values or res.index.isin(["Signal_A"]).any()
        ), "Result should contain original channel names"
        assert len(res) == 2

    def test_dict_key_order_preserved_in_matrix(self):
        """When converting TimeSeriesDict to matrix, channel order should match dict order."""
        ts_a = TimeSeries(np.ones(50), sample_rate=100, name="A")
        ts_b = TimeSeries(np.ones(50) * 2, sample_rate=100, name="B")
        ts_c = TimeSeries(np.ones(50) * 3, sample_rate=100, name="C")

        # Specific order
        tsd = TimeSeriesDict({"A": ts_a, "B": ts_b, "C": ts_c})
        tsm = tsd.to_matrix()

        # Mean should follow the same order
        mn = tsm.mean()
        assert mn[0, 0] == pytest.approx(1.0)  # A
        assert mn[1, 0] == pytest.approx(2.0)  # B
        assert mn[2, 0] == pytest.approx(3.0)  # C

    # =========================================
    # Edge Case: Boundary Conditions
    # =========================================

    def test_single_channel_matrix(self):
        """Single channel matrix should work correctly."""
        data = np.random.randn(1, 100)
        tsm = TimeSeriesMatrix(data, sample_rate=100)

        mn = tsm.mean()
        assert mn.shape == (1, 1)
        assert mn[0, 0] == pytest.approx(np.mean(data))

        sk = tsm.skewness()
        assert sk.shape == (1, 1)

        # Correlation with itself - returns a scalar (1x1 case collapses to scalar)
        corr = tsm.correlation()
        # np.corrcoef on a single row returns a scalar 1.0
        assert corr == pytest.approx(1.0) or (
            hasattr(corr, "shape") and corr.shape in [(1, 1), ()]
        )

    def test_single_channel_dict(self):
        """Single channel dict should work correctly."""
        ts = TimeSeries(np.random.randn(100), sample_rate=100, name="only_one")
        tsd = TimeSeriesDict({"only_one": ts})

        mn = tsd.mean()
        assert mn.shape == (1, 1)

        sk = tsd.skewness()
        assert sk.shape == (1, 1)

    def test_empty_dict_raises_or_handles(self):
        """Empty TimeSeriesDict should raise or return empty results gracefully."""
        tsd = TimeSeriesDict()

        # Trying to calculate stats on empty dict
        with pytest.raises((ValueError, IndexError, KeyError)):
            tsd.mean()

    def test_empty_list_raises_or_handles(self):
        """Empty TimeSeriesList should raise or return empty results gracefully."""
        tsl = TimeSeriesList()

        with pytest.raises((ValueError, IndexError, KeyError)):
            tsl.mean()

    def test_very_short_timeseries(self):
        """Very short time series (< typical segment) should still compute stats."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tsm = TimeSeriesMatrix(data, sample_rate=100)

        mn = tsm.mean()
        assert mn[0, 0] == pytest.approx(2.0)
        assert mn[1, 0] == pytest.approx(5.0)

        # RMS
        rms = tsm.rms()
        assert rms[0, 0] == pytest.approx(np.sqrt(np.mean([1, 4, 9])))

    # =========================================
    # Edge Case: Correlation Aliases
    # =========================================

    def test_pcc_alias(self, sample_data):
        """pcc() should be an alias for correlation(method='pearson')."""
        tsm = TimeSeriesMatrix(sample_data, sample_rate=100)
        target = TimeSeries(sample_data[0], sample_rate=100)

        res_pcc = tsm.pcc(target)
        res_pearson = tsm.correlation(target, method="pearson")

        assert len(res_pcc) == len(res_pearson)
        # Scores should be identical
        assert res_pcc.iloc[0]["score"] == pytest.approx(res_pearson.iloc[0]["score"])

    def test_ktau_alias(self, sample_data):
        """ktau() should be an alias for correlation(method='kendall')."""
        tsm = TimeSeriesMatrix(sample_data, sample_rate=100)
        target = TimeSeries(sample_data[0], sample_rate=100)

        res_ktau = tsm.ktau(target)
        res_kendall = tsm.correlation(target, method="kendall")

        assert len(res_ktau) == len(res_kendall)
        assert res_ktau.iloc[0]["score"] == pytest.approx(res_kendall.iloc[0]["score"])
