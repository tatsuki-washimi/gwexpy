"""Tests for ScalarField signal processing utilities (PSD, XCorr, coherence)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose
from scipy.signal import correlate, welch

from gwexpy.fields import ScalarField
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList
from gwexpy.timeseries import TimeSeries


@pytest.fixture
def sine_field():
    """Deterministic 2-point field with a known phase lead on x=1."""

    dt = 0.01 * u.s
    nt = 128
    times = np.arange(nt) * dt
    x = np.arange(2) * 1.0 * u.m

    base = np.sin(2 * np.pi * 5.0 * times.to_value(u.s))
    data = np.zeros((nt, 2, 1, 1))
    data[:, 0, 0, 0] = base
    data[:, 1, 0, 0] = np.roll(base, -3)  # point x=1 leads by 3 samples

    return ScalarField(
        data,
        unit=u.V,
        axis0=times,
        axis1=x,
        axis2=np.array([0.0]) * u.m,
        axis3=np.array([0.0]) * u.m,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
        space_domain="real",
    )


class TestComputePsd:
    def test_single_point_detects_peak_frequency(self, sine_field):
        psd = sine_field.compute_psd((0 * u.m, 0 * u.m, 0 * u.m), nperseg=64)

        assert isinstance(psd, FrequencySeries)
        peak_freq = psd.frequencies.value[np.argmax(psd.value)]
        # Bin resolution is fs/nperseg = 1.5625 Hz; accept nearest-bin peak.
        assert_allclose(peak_freq, 5.0, atol=0.8)
        assert psd.frequencies.unit == u.Hz
        assert psd.unit == u.V**2 / u.Hz

    def test_multiple_points_returns_list(self, sine_field):
        psd = sine_field.compute_psd(
            [(0 * u.m, 0 * u.m, 0 * u.m), (1 * u.m, 0 * u.m, 0 * u.m)],
            nperseg=64,
        )

        assert isinstance(psd, FrequencySeriesList)
        assert len(psd) == 2
        assert [p.name for p in psd] == ["point_0", "point_1"]

    def test_region_average_matches_manual_welch(self, sine_field):
        psd_region = sine_field.compute_psd({"x": slice(None)}, nperseg=64)

        averaged = np.mean(sine_field.value, axis=(1, 2, 3))
        freqs, expected = welch(averaged, fs=1.0 / 0.01, nperseg=64)

        assert_allclose(psd_region.frequencies.value, freqs)
        assert_allclose(psd_region.value, expected)

    def test_irregular_time_axis_raises(self):
        times = np.array([0.0, 1.0, 2.0, 4.0]) * u.s
        field = ScalarField(
            np.ones((4, 1, 1, 1)),
            axis0=times,
            axis1=np.array([0.0]) * u.m,
            axis2=np.array([0.0]) * u.m,
            axis3=np.array([0.0]) * u.m,
            axis0_domain="time",
        )

        with pytest.raises(ValueError, match="regularly spaced"):
            field.compute_psd((0 * u.m, 0 * u.m, 0 * u.m))

    def test_wrong_domain_raises(self):
        field = ScalarField(
            np.ones((8, 1, 1, 1)),
            axis0=np.arange(8) * 0.1 * u.Hz,
            axis0_domain="frequency",
        )

        with pytest.raises(ValueError, match="axis0_domain='time'"):
            field.compute_psd((0 * u.m, 0 * u.m, 0 * u.m))

    def test_time_axis_too_short(self):
        field = ScalarField(
            np.ones((1, 1, 1, 1)),
            axis0=np.array([0.0]) * u.s,
            axis0_domain="time",
        )

        with pytest.raises(ValueError, match="at least 2 points"):
            field.compute_psd((0 * u.m, 0 * u.m, 0 * u.m))


class TestFreqSpaceMap:
    def test_shape_and_units(self, sine_field):
        fs_map = sine_field.freq_space_map("x", nperseg=64)

        assert fs_map.axis0_domain == "frequency"
        assert fs_map.axis_names == ("f", "x", "y", "z")
        assert fs_map.shape[1] == sine_field.shape[1]
        assert fs_map.shape[0] == len(fs_map._axis0_index)
        assert fs_map.unit == u.V**2 / u.Hz

    def test_requires_fixed_other_axes(self):
        field = ScalarField(
            np.ones((8, 2, 2, 1)),
            axis0=np.arange(8) * 0.1 * u.s,
            axis1=np.arange(2) * 1.0 * u.m,
            axis2=np.arange(2) * 1.0 * u.m,
            axis3=np.array([0.0]) * u.m,
            axis_names=["t", "x", "y", "z"],
            axis0_domain="time",
            space_domain="real",
        )

        with pytest.raises(ValueError, match="Specify its value in 'at'"):
            field.freq_space_map("x")


class TestComputeXcorr:
    def test_peak_lag_matches_correlate(self, sine_field):
        result = sine_field.compute_xcorr(
            (0 * u.m, 0 * u.m, 0 * u.m),
            (1 * u.m, 0 * u.m, 0 * u.m),
            mode="full",
            detrend=True,
            normalize=True,
        )

        assert isinstance(result, TimeSeries)
        data_a = sine_field.value[:, 0, 0, 0]
        data_b = sine_field.value[:, 1, 0, 0]
        expected = correlate(
            data_a - np.mean(data_a), data_b - np.mean(data_b), mode="full"
        )
        lags = np.arange(-(len(data_a) - 1), len(data_a)) * 0.01

        peak_idx = np.argmax(np.abs(expected))
        expected_lag = lags[peak_idx]

        result_peak_idx = np.argmax(np.abs(result.value))
        assert_allclose(result.times.value[result_peak_idx], expected_lag)
        assert result.unit == u.dimensionless_unscaled
        assert np.max(np.abs(result.value)) <= 1.0 + 1e-12

    def test_time_axis_too_short_raises(self):
        field = ScalarField(
            np.ones((1, 1, 1, 1)),
            axis0=np.array([0.0]) * u.s,
            axis0_domain="time",
        )

        with pytest.raises(ValueError, match="at least 2 points"):
            field.compute_xcorr(
                (0 * u.m, 0 * u.m, 0 * u.m), (0 * u.m, 0 * u.m, 0 * u.m)
            )


class TestTimeDelayMap:
    def test_map_shape_metadata_and_values(self, sine_field):
        delay_map = sine_field.time_delay_map(
            (0 * u.m, 0 * u.m, 0 * u.m),
            plane="xy",
            stride=1,
        )

        assert delay_map.axis0_domain == "time"
        assert delay_map.axis_names == ("t", "x", "y", "z")
        assert delay_map.shape[1] == sine_field.shape[1]
        assert delay_map.unit == u.s

        xcorr = sine_field.compute_xcorr(
            (0 * u.m, 0 * u.m, 0 * u.m),
            (1 * u.m, 0 * u.m, 0 * u.m),
            mode="full",
            detrend=True,
            normalize=True,
        )
        expected_lag = xcorr.times.value[np.argmax(np.abs(xcorr.value))]

        assert_allclose(delay_map.value[0, 1, 0, 0], expected_lag)


class TestCoherenceMap:
    def test_band_averaged_map_shape_and_range(self, sine_field):
        coh_map = sine_field.coherence_map(
            (0 * u.m, 0 * u.m, 0 * u.m),
            plane="xy",
            band=(1.0 * u.Hz, 20.0 * u.Hz),
            nperseg=64,
        )

        assert coh_map.axis0_domain == "time"
        assert coh_map.axis_names == ("t", "x", "y", "z")
        assert coh_map.shape[0] == 1
        assert coh_map.unit == u.dimensionless_unscaled
        assert np.all((coh_map.value >= 0.0) & (coh_map.value <= 1.0))
        assert coh_map.shape[1] == sine_field.shape[1]

    def test_frequency_resolved_map(self, sine_field):
        coh_map = sine_field.coherence_map(
            (0 * u.m, 0 * u.m, 0 * u.m),
            plane="xy",
            band=None,
            nperseg=64,
        )

        assert coh_map.axis0_domain == "frequency"
        assert coh_map.axis_names[0] == "f"
        assert coh_map.shape[1] == sine_field.shape[1]
        assert coh_map.unit == u.dimensionless_unscaled
        assert np.all(coh_map.value >= -1e-9)
        assert np.all(coh_map.value <= 1.0 + 1e-6)
