"""Tests for gwexpy/plot/defaults.py."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from astropy import units as u

from gwexpy.plot.defaults import (
    _format_unit_label,
    _is_linear_unit_or_name,
    calculate_default_figsize,
    determine_clabel,
    determine_geometry_and_separate,
    determine_norm,
    determine_xlabel,
    determine_ylabel,
    determine_ylim,
    determine_yscale,
    determine_xscale,
)


# ---------------------------------------------------------------------------
# calculate_default_figsize
# ---------------------------------------------------------------------------

class TestCalculateDefaultFigsize:
    def test_geometry_overrides_nrow_ncol(self):
        w, h = calculate_default_figsize((2, 3), nrow=1, ncol=1)
        # geometry=(2,3) → ncol=3, nrow=2
        assert w == min(28, 9 * 3)
        assert h == min(24, 5 * 2)

    def test_no_geometry_uses_nrow_ncol(self):
        w, h = calculate_default_figsize(None, nrow=3, ncol=2)
        assert w == min(28, 9 * 2)
        assert h == min(24, 5 * 3)

    def test_max_width_clamped(self):
        w, h = calculate_default_figsize(None, nrow=1, ncol=10)
        assert w == 28

    def test_max_height_clamped(self):
        w, h = calculate_default_figsize(None, nrow=10, ncol=1)
        assert h == 24


# ---------------------------------------------------------------------------
# _is_linear_unit_or_name
# ---------------------------------------------------------------------------

class TestIsLinearUnitOrName:
    def test_deg_unit(self):
        assert _is_linear_unit_or_name(u.deg, "") is True

    def test_rad_unit(self):
        assert _is_linear_unit_or_name(u.rad, "") is True

    def test_dB_unit(self):
        assert _is_linear_unit_or_name(u.dB, "") is True

    def test_phase_in_name(self):
        assert _is_linear_unit_or_name(u.m, "phase response") is True

    def test_delay_in_name(self):
        assert _is_linear_unit_or_name(u.m, "group delay") is True

    def test_angle_in_name(self):
        assert _is_linear_unit_or_name(u.m, "angle") is True

    def test_non_linear_unit(self):
        assert _is_linear_unit_or_name(u.m, "amplitude") is False

    def test_none_unit(self):
        assert _is_linear_unit_or_name(None, "") is False


# ---------------------------------------------------------------------------
# _format_unit_label
# ---------------------------------------------------------------------------

class TestFormatUnitLabel:
    def test_none_returns_none(self):
        assert _format_unit_label(None) is None

    def test_dimensionless_returns_none(self):
        assert _format_unit_label(u.dimensionless_unscaled) is None

    def test_meter_unit(self):
        label = _format_unit_label(u.m)
        assert label is not None
        assert "m" in label

    def test_hz_unit(self):
        label = _format_unit_label(u.Hz)
        assert label is not None


# ---------------------------------------------------------------------------
# determine_xscale
# ---------------------------------------------------------------------------

class TestDetermineXscale:
    def test_current_value_respected(self):
        assert determine_xscale([], current_value="log") == "log"

    def test_empty_list_returns_none(self):
        assert determine_xscale([]) is None

    def test_frequency_series_large(self):
        from gwpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(512), df=1.0)
        result = determine_xscale([fs])
        assert result == "log"

    def test_frequency_series_small(self):
        from gwpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(10), df=1.0)
        result = determine_xscale([fs])
        assert result is None

    def test_spectrogram_returns_auto_gps(self):
        from gwpy.spectrogram import Spectrogram
        data = np.ones((10, 5))
        sg = Spectrogram(data, t0=0, dt=1, f0=0, df=1)
        result = determine_xscale([sg])
        assert result == "auto-gps"


# ---------------------------------------------------------------------------
# determine_yscale
# ---------------------------------------------------------------------------

class TestDetermineYscale:
    def test_current_value_respected(self):
        assert determine_yscale([], current_value="linear") == "linear"

    def test_empty_list_returns_none(self):
        assert determine_yscale([]) is None

    def test_spectrogram_returns_log(self):
        from gwpy.spectrogram import Spectrogram
        data = np.ones((10, 5))
        sg = Spectrogram(data, t0=0, dt=1, f0=0, df=1)
        result = determine_yscale([sg])
        assert result == "log"

    def test_frequency_series_non_linear_unit(self):
        from gwpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(10), df=1.0, unit="m")
        result = determine_yscale([fs])
        assert result == "log"

    def test_frequency_series_linear_unit(self):
        from gwpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(10), df=1.0, unit="deg")
        result = determine_yscale([fs])
        assert result is None


# ---------------------------------------------------------------------------
# determine_norm
# ---------------------------------------------------------------------------

class TestDetermineNorm:
    def test_current_value_respected(self):
        assert determine_norm([], current_value="linear") == "linear"

    def test_empty_list_returns_none(self):
        assert determine_norm([]) is None

    def test_spectrogram_returns_log(self):
        from gwpy.spectrogram import Spectrogram
        data = np.ones((10, 5))
        sg = Spectrogram(data, t0=0, dt=1, f0=0, df=1)
        result = determine_norm([sg])
        assert result == "log"

    def test_non_spectrogram_returns_none(self):
        from gwpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(10), df=1.0)
        assert determine_norm([fs]) is None


# ---------------------------------------------------------------------------
# determine_geometry_and_separate
# ---------------------------------------------------------------------------

class TestDetermineGeometryAndSeparate:
    def test_empty_list_unchanged(self):
        sep, geom = determine_geometry_and_separate([])
        assert sep is None
        assert geom is None

    def test_spectrogram_sets_separate_true(self):
        from gwpy.spectrogram import Spectrogram
        data = np.ones((10, 5))
        sg = Spectrogram(data, t0=0, dt=1, f0=0, df=1)
        sep, geom = determine_geometry_and_separate([sg])
        assert sep is True

    def test_current_geometry_not_overridden(self):
        from gwpy.spectrogram import Spectrogram
        data = np.ones((10, 5))
        sg = Spectrogram(data, t0=0, dt=1, f0=0, df=1)
        sep, geom = determine_geometry_and_separate([sg], geometry=(2, 2))
        assert geom == (2, 2)


# ---------------------------------------------------------------------------
# determine_xlabel
# ---------------------------------------------------------------------------

class TestDetermineXlabel:
    def test_current_value_respected(self):
        assert determine_xlabel([], current_value="Time [s]") == "Time [s]"

    def test_empty_list_returns_none(self):
        assert determine_xlabel([]) is None

    def test_frequency_series_returns_frequency_label(self):
        from gwpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(10), df=1.0)
        label = determine_xlabel([fs])
        assert label is not None
        assert "Freq" in label or "Hz" in label or "freq" in label.lower()


# ---------------------------------------------------------------------------
# determine_ylabel
# ---------------------------------------------------------------------------

class TestDetermineYlabel:
    def test_current_value_respected(self):
        assert determine_ylabel([], current_value="Amplitude") == "Amplitude"

    def test_empty_list_returns_none(self):
        assert determine_ylabel([]) is None

    def test_frequency_series_with_unit(self):
        from gwpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(10), df=1.0, unit="m")
        label = determine_ylabel([fs])
        assert label is not None
        assert "m" in label

    def test_spectrogram_ylabel(self):
        from gwpy.spectrogram import Spectrogram
        data = np.ones((10, 5))
        sg = Spectrogram(data, t0=0, dt=1, f0=0, df=1)
        label = determine_ylabel([sg])
        assert label is not None


# ---------------------------------------------------------------------------
# determine_clabel
# ---------------------------------------------------------------------------

class TestDetermineClabel:
    def test_current_value_respected(self):
        assert determine_clabel([], current_value="Power") == "Power"

    def test_empty_list_returns_none(self):
        assert determine_clabel([]) is None

    def test_spectrogram_returns_unit_label(self):
        from gwpy.spectrogram import Spectrogram
        data = np.ones((10, 5))
        sg = Spectrogram(data, t0=0, dt=1, f0=0, df=1, unit="m")
        label = determine_clabel([sg])
        assert label is not None
        assert "m" in label

    def test_non_spectrogram_returns_none(self):
        from gwpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(10), df=1.0)
        assert determine_clabel([fs]) is None
