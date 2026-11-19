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


class TestRootScaleInvariance:
    def test_is_regular_spacing_handles_1e_20_scale(self):
        from gwexpy.interop.root_ import _is_regular_spacing

        assert _is_regular_spacing(np.diff(np.array([1e-20, 2e-20, 3e-20, 4e-20])))
        assert not _is_regular_spacing(np.diff(np.array([1e-20, 2e-20, 4e-20, 8e-20])))
        assert not _is_regular_spacing(np.array([0.0, 0.0]))
        assert not _is_regular_spacing(np.array([1.0, np.nan]))

    def test_root_roundtrip_1e_20_irregular_and_regular_axes(self):
        ROOT = pytest.importorskip("ROOT")
        from gwexpy.frequencyseries import FrequencySeries
        from gwexpy.spectrogram import Spectrogram

        ROOT.gROOT.SetBatch(True)

        # Some tiny axes cannot be represented by ROOT bin centers exactly.
        irr_x = np.array([1e-20, 2e-20, 4e-20, 8e-20])
        fs_irr = FrequencySeries(np.arange(1.0, 5.0), frequencies=irr_x, name="fs_irr")
        with pytest.raises(ValueError, match="exact sample coordinates"):
            fs_irr.to_th1d()

        # TGraph preserves exact irregular 1e-20 coordinates without linearization
        g_irr = fs_irr.to_tgraph()
        fs_g_back = FrequencySeries.from_root(g_irr)
        np.testing.assert_allclose(
            fs_g_back.frequencies.value, irr_x, rtol=1e-12, atol=0.0
        )

        # Even apparently regular 1e-20 spacing can lose an ULP in ROOT bins.
        reg_x = np.array([1e-20, 2e-20, 3e-20, 4e-20])
        fs_reg = FrequencySeries(np.arange(1.0, 5.0), frequencies=reg_x, name="fs_reg")
        with pytest.raises(ValueError, match="exact sample coordinates"):
            fs_reg.to_th1d()

        # Spectrogram to_th2d with 1e-20 irregular frequency axis
        sg = Spectrogram(
            np.ones((3, 4)),
            times=np.array([0.0, 1e-20, 3e-20]),
            frequencies=irr_x,
            name="sg_tiny",
        )
        with pytest.raises(ValueError, match="exact sample coordinates"):
            sg.to_th2d()

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
