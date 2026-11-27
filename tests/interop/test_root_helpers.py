"""Tests for ROOT interop helper functions (no ROOT required)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.interop.root_ import _extract_error_array, _get_label
from gwexpy.timeseries import TimeSeries

# ---------------------------------------------------------------------------
# _get_label
# ---------------------------------------------------------------------------


class TestGetLabel:
    def test_name_with_unit(self):
        obj = type("O", (), {"name": "signal"})()
        label = _get_label(obj, u.m)
        assert label == "signal [m]"

    def test_name_without_unit(self):
        obj = type("O", (), {"name": "signal"})()
        label = _get_label(obj, None)
        assert label == "signal"

    def test_no_name_uses_default(self):
        obj = type("O", (), {})()
        label = _get_label(obj, None, default_name="xaxis")
        assert label == "xaxis"

    def test_empty_name_uses_default(self):
        obj = type("O", (), {"name": ""})()
        label = _get_label(obj, None, default_name="yaxis")
        assert label == "yaxis"

    def test_none_name_uses_default(self):
        obj = type("O", (), {"name": None})()
        label = _get_label(obj, u.Hz, default_name="freq")
        assert label == "freq [Hz]"

    def test_dimensionless_unit_string(self):
        obj = type("O", (), {"name": "phase"})()
        label = _get_label(obj, u.rad)
        assert "phase" in label
        assert "rad" in label


# ---------------------------------------------------------------------------
# _extract_error_array
# ---------------------------------------------------------------------------


class _FakeSeries:
    """Minimal gwpy-like Series with .value and .xindex."""

    def __init__(self, data):
        self.value = np.asarray(data)
        self.xindex = self.value
        self.unit = u.m
        self.shape = self.value.shape

    def __len__(self):
        return len(self.value)


class TestExtractErrorArray:
    def test_plain_array(self):
        series = _FakeSeries(np.ones(5))
        error = np.ones(5) * 0.1
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 0.1)

    def test_plain_list(self):
        series = _FakeSeries(np.ones(5))
        result = _extract_error_array(series, [0.1] * 5)
        assert result.shape == (5,)

    def test_plain_array_shape_mismatch_raises(self):
        series = _FakeSeries(np.ones(5))
        with pytest.raises(ValueError, match="shape"):
            _extract_error_array(series, np.ones(3))

    def test_gwpy_series_error(self):
        """Error with .value and .xindex attributes (gwpy-like)."""
        series = _FakeSeries(np.ones(5))
        error = _FakeSeries(np.ones(5) * 0.2)
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 0.2)

    def test_gwpy_series_length_mismatch_raises(self):
        series = _FakeSeries(np.ones(5))
        error = _FakeSeries(np.ones(3))
        with pytest.raises(ValueError, match="length"):
            _extract_error_array(series, error)

    def test_astropy_quantity_same_unit(self):
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(5) * 0.1, u.m)
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 0.1)

    def test_astropy_quantity_convertible_unit(self):
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(5) * 100, u.cm)  # 100 cm = 1 m
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 1.0)

    def test_astropy_quantity_incompatible_unit_raises(self):
        """An incompatible error unit is never used as a raw magnitude (#773)."""
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(5) * 0.5, u.s)
        with pytest.raises(u.UnitConversionError, match="error.*'s'.*'m'"):
            _extract_error_array(series, error)

    def test_dimensionless_quantity_for_unitful_series_raises(self):
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(5) * 0.5)
        with pytest.raises(u.UnitConversionError):
            _extract_error_array(series, error)

    def test_quantity_for_dimensionless_series(self):
        series = _FakeSeries(np.ones(5))
        series.unit = u.dimensionless_unscaled
        result = _extract_error_array(series, u.Quantity(np.ones(5) * 0.5))
        np.testing.assert_array_equal(result, 0.5)
        with pytest.raises(u.UnitConversionError):
            _extract_error_array(series, u.Quantity(np.ones(5), u.m))

    def test_quantity_for_series_without_unit_uses_value(self):
        series = _FakeSeries(np.ones(5))
        series.unit = None
        result = _extract_error_array(series, u.Quantity(np.ones(5) * 0.5, u.s))
        np.testing.assert_array_equal(result, 0.5)

    def test_error_series_convertible_unit_is_scaled(self):
        series = _FakeSeries(np.ones(5))
        error = _FakeSeries(np.ones(5) * 100)
        error.unit = u.cm
        result = _extract_error_array(series, error)
        np.testing.assert_allclose(result, 1.0, rtol=1e-12)

    def test_error_series_incompatible_unit_raises(self):
        series = _FakeSeries(np.ones(5))
        error = _FakeSeries(np.ones(5) * 0.5)
        error.unit = u.s
        with pytest.raises(u.UnitConversionError, match="error.*'s'.*'m'"):
            _extract_error_array(series, error)

    def test_error_series_without_declared_unit_uses_value(self):
        """A dimensionless error series carries no unit claim to contradict."""
        series = _FakeSeries(np.ones(5))
        for unit in (None, u.dimensionless_unscaled):
            error = _FakeSeries(np.ones(5) * 0.5)
            error.unit = unit
            np.testing.assert_array_equal(_extract_error_array(series, error), 0.5)

    def test_timeseries_error_units(self):
        data = TimeSeries(np.ones(4), t0=0, dt=1, unit="m")
        scaled = _extract_error_array(
            data, TimeSeries(np.full(4, 5.0), t0=0, dt=1, unit="mm")
        )
        np.testing.assert_allclose(scaled, 0.005, rtol=1e-12)
        with pytest.raises(u.UnitConversionError):
            _extract_error_array(data, TimeSeries(np.ones(4), t0=0, dt=1, unit="s"))
        with pytest.raises(u.UnitConversionError):
            _extract_error_array(data, np.full(4, 10.0) * u.s)

    def test_tiny_strain_scale_is_preserved(self):
        data = TimeSeries(np.full(4, 1e-21), t0=0, dt=1, unit="m")
        result = _extract_error_array(data, np.full(4, 2e-19) * u.cm)
        np.testing.assert_allclose(result, 2e-21, rtol=1e-12)

    def test_astropy_quantity_shape_mismatch_raises(self):
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(3), u.m)
        with pytest.raises(ValueError, match="shape"):
            _extract_error_array(series, error)

    def test_timeseries_as_error(self):
        ts_series = TimeSeries(np.ones(10), t0=0, dt=0.01)
        ts_error = TimeSeries(np.ones(10) * 0.05, t0=0, dt=0.01)
        result = _extract_error_array(ts_series, ts_error)
        assert result.shape == (10,)


#: 1 ns alternating jitter on a 1 us grid: GWpy's absolute tolerance calls it
#: regular (#887), but its first interval misplaces every odd sample.
_JITTER_887 = np.arange(100) * 1e-6 + np.where(np.arange(100) % 2, 1e-9, 0.0)


class TestRootScaleInvariance:
    def test_root_roundtrip_1e_20_irregular_and_regular_axes(self):
        ROOT = pytest.importorskip("ROOT")
        from gwexpy.frequencyseries import FrequencySeries
        from gwexpy.spectrogram import Spectrogram

        ROOT.gROOT.SetBatch(True)

        # TGraph preserves exact irregular 1e-20 coordinates without linearization
        irr_x = np.array([1e-20, 2e-20, 4e-20, 8e-20])
        fs_irr = FrequencySeries(np.arange(1.0, 5.0), frequencies=irr_x, name="fs_irr")
        fs_g_back = FrequencySeries.from_root(fs_irr.to_tgraph())
        np.testing.assert_array_equal(fs_g_back.frequencies.value, irr_x)

        # A uniform 1e-20 axis uses fixed-width bins, at any scale.
        reg_x = np.array([1e-20, 2e-20, 3e-20, 4e-20])
        fs_reg = FrequencySeries(np.arange(1.0, 5.0), frequencies=reg_x, name="fs_reg")
        hist = fs_reg.to_th1d()
        assert not hist.GetXaxis().IsVariableBinSize()
        assert hist.GetXaxis().GetBinWidth(1) == pytest.approx(1e-20, rel=1e-12)

        # A non-uniform axis whose samples cannot all be bin centres is rejected
        # instead of being moved to midpoint-derived centres.
        sg = Spectrogram(
            np.ones((4, 4)),
            times=np.array([0.0, 1e-20, 11e-20, 12e-20]),
            frequencies=irr_x,
            name="sg_tiny",
        )
        with pytest.raises(ValueError, match="exact sample coordinates"):
            sg.to_th2d()

    def test_uniform_series_keeps_fixed_width_binning(self):
        ROOT = pytest.importorskip("ROOT")
        from gwexpy.frequencyseries import FrequencySeries

        ROOT.gROOT.SetBatch(True)
        cases = [
            TimeSeries(np.arange(4096.0), t0=1264316116, sample_rate=4096),
            TimeSeries(np.arange(50.0), t0=0.1, dt=0.1),
            FrequencySeries(np.arange(300.0), f0=10, df=1 / 3),
            FrequencySeries(np.arange(1000.0), frequencies=np.linspace(1, 10, 1000)),
        ]
        for series in cases:
            hist = series.to_th1d()
            axis = hist.GetXaxis()
            assert not axis.IsVariableBinSize()
            x = series.xindex.value
            centers = np.array([axis.GetBinCenter(i + 1) for i in range(x.size)])
            # Uniform bins are defined by their edges; ROOT derives centres.
            np.testing.assert_allclose(centers, x, rtol=0, atol=1e-12 * abs(x).max())
            back = type(series).from_root(hist)
            assert back.xindex.regular
            assert back.dx.value == pytest.approx(series.dx.value, rel=1e-12)

    def test_gwpy_regular_nonaffine_axis_is_not_linearized(self):
        ROOT = pytest.importorskip("ROOT")

        ROOT.gROOT.SetBatch(True)
        ts = TimeSeries(np.ones(_JITTER_887.size), times=_JITTER_887, name="jit")
        assert ts.xindex.regular
        # Not fixed-width bins from the first interval: variable bins centred
        # on the jittered samples.
        hist = ts.to_th1d()
        axis = hist.GetXaxis()
        assert axis.IsVariableBinSize()
        centers = np.array([axis.GetBinCenter(i + 1) for i in range(_JITTER_887.size)])
        np.testing.assert_allclose(centers, _JITTER_887, rtol=0, atol=1e-21)
        back = TimeSeries.from_root(ts.to_tgraph())
        np.testing.assert_array_equal(back.times.value, _JITTER_887)

    @pytest.mark.parametrize(
        "f",
        [
            np.geomspace(1, 1000, 100),
            np.logspace(0, 3, 30),
            np.geomspace(0.1, 1000, 30),
            np.array([1.0, 1.5, 1.9, 2.4, 2.8, 3.3]),
        ],
        ids=["geomspace100", "logspace30", "geomspace30", "alternating"],
    )
    def test_irregular_series_keeps_variable_bin_centres(self, f):
        ROOT = pytest.importorskip("ROOT")
        from gwexpy.frequencyseries import FrequencySeries
        from gwexpy.spectrogram import Spectrogram

        ROOT.gROOT.SetBatch(True)
        fs = FrequencySeries(np.arange(float(f.size)), frequencies=f, name="log")
        hist = fs.to_th1d()
        axis = hist.GetXaxis()
        assert axis.IsVariableBinSize()
        centers = np.array([axis.GetBinCenter(i + 1) for i in range(f.size)])
        # Centres are derived from stored edges: equal up to edge round-off.
        np.testing.assert_allclose(centers, f, rtol=4 * np.finfo(float).eps, atol=0)
        back = FrequencySeries.from_root(hist).frequencies.value
        np.testing.assert_array_equal(back, centers)

        spec = Spectrogram(np.ones((3, f.size)), times=[0.0, 1.0, 2.0], frequencies=f)
        hist2 = spec.to_th2d()
        ycent = np.array([hist2.GetYaxis().GetBinCenter(i + 1) for i in range(f.size)])
        np.testing.assert_allclose(ycent, f, rtol=4 * np.finfo(float).eps, atol=0)

    def test_series_without_centring_bins_raises(self):
        ROOT = pytest.importorskip("ROOT")

        ROOT.gROOT.SetBatch(True)
        # Spacing 1 then 10 then 1: no increasing edges centre all four bins.
        ts = TimeSeries(np.ones(4), times=[0.0, 1.0, 11.0, 12.0])
        with pytest.raises(ValueError, match="no increasing bin edges"):
            ts.to_th1d()

    def test_histogram_regularity_follows_bin_edges(self):
        ROOT = pytest.importorskip("ROOT")

        ROOT.gROOT.SetBatch(True)
        # Fixed-width binning: affine series even where ROOT's derived centres
        # differ from x0 + i * dx by round-off.
        fixed = ROOT.TH1D("u7_fixed", "", 1000, -3.3, 7.7)
        ts = TimeSeries.from_root(fixed)
        assert ts.x0.value == fixed.GetXaxis().GetBinCenter(1)
        assert ts.dt.value == fixed.GetXaxis().GetBinWidth(1)
        assert ts.xindex.size == 1000

        # Variable-width binning: the bin centres are the explicit coordinates.
        edges = np.array([0.0, 1.0, 3.0, 7.0, 15.0])
        variable = ROOT.TH1D("u7_variable", "", 4, edges)
        ts_var = TimeSeries.from_root(variable)
        np.testing.assert_array_equal(ts_var.times.value, [0.5, 2.0, 5.0, 11.0])

    def test_near_regular_root_bins_preserve_exact_centers(self):
        ROOT = pytest.importorskip("ROOT")
        from gwexpy.frequencyseries import FrequencySeries
        from gwexpy.spectrogram import Spectrogram

        ROOT.gROOT.SetBatch(True)
        x = np.array([0.0, 1.0, 2.0 + 1e-7, 3.0 + 2e-7])
        fs = FrequencySeries(np.arange(4.0), frequencies=x, name="near_regular")
        hist = fs.to_th1d()
        centers = np.array([hist.GetXaxis().GetBinCenter(i + 1) for i in range(4)])
        np.testing.assert_array_equal(centers, x)
        np.testing.assert_array_equal(
            FrequencySeries.from_root(hist).frequencies.value, x
        )

        spec = Spectrogram(
            np.ones((4, 4)), times=x, frequencies=x, name="near_regular_spec"
        )
        hist2 = spec.to_th2d()
        times = np.array([hist2.GetXaxis().GetBinCenter(i + 1) for i in range(4)])
        freqs = np.array([hist2.GetYaxis().GetBinCenter(i + 1) for i in range(4)])
        np.testing.assert_array_equal(times, x)
        np.testing.assert_array_equal(freqs, x)
