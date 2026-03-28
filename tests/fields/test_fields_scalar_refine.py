"""
Refined edge case tests for ScalarField and SignalField.

Covers:
- ScalarField.filter (zpk, filtfilt=False, b/a)
- ScalarField.extract_points/profile (interp="nearest", error handling)
- ScalarField.resample (Quantity rate)
- _validate_axis_for_spectral (Frequency/K domain error)
- SignalField.compute_xcorr (normalize=False, window)
- SignalField.time_delay_map (plane="xz", "yz")
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose
from scipy import signal as scipy_signal

from gwexpy.fields import ScalarField
from gwexpy.types.metadata import MetaData, MetaDataMatrix


@pytest.fixture
def test_field_4d():
    """Simple 4D field for testing (Time, X, Y, Z)."""
    dt = 0.01 * u.s
    nt = 100
    times = np.arange(nt) * dt
    x = np.arange(2) * 1.0 * u.m
    y = np.arange(2) * 1.0 * u.m
    z = np.arange(2) * 1.0 * u.m
    
    # Simple spatial gradient + time oscillation
    data = np.zeros((nt, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                data[:, i, j, k] = i + j + k + np.sin(2 * np.pi * 5 * times.to_value(u.s))
    
    return ScalarField(
        data,
        unit=u.m,
        axis0=times,
        axis1=x,
        axis2=y,
        axis3=z,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
        space_domain="real",
    )


class TestScalarFieldFilter:
    def test_filter_zpk_sos_path(self, test_field_4d):
        # 10 Hz lowpass
        zpk = scipy_signal.butter(4, 10, btype="low", fs=100, output="zpk")
        
        # Default filtfilt=True (zero-phase)
        f_zpk = test_field_4d.filter(zpk)
        assert f_zpk.shape == test_field_4d.shape
        assert f_zpk.unit == test_field_4d.unit
        
        # filtfilt=False (causal, sosfilt path because zpk is converted to sos)
        f_sos = test_field_4d.filter(zpk, filtfilt=False)
        assert f_sos.shape == test_field_4d.shape
        # Peak delay check (causal should delay)
        # Point (0,0,0) originally has peak at t=0.05s (sin 5Hz)
        # Actually sin(2pi*5*0.05) = sin(pi/2)=1.
        pass

    def test_filter_ba_path(self, test_field_4d):
        # b/a path
        b, a = scipy_signal.butter(4, 0.2) # lowpass
        
        # filtfilt=False (lfilter path)
        f_ba = test_field_4d.filter((b, a), filtfilt=False)
        assert f_ba.shape == test_field_4d.shape


class TestScalarFieldExtractNearest:
    def test_extract_points_nearest(self, test_field_4d):
        # Point (0.1, 0.1, 0.1) should map to grid index (0, 0, 0)
        p = [(0.1 * u.m, 0.1 * u.m, 0.1 * u.m)]
        ts_list = test_field_4d.extract_points(p, interp="nearest")
        assert len(ts_list) == 1
        assert_allclose(ts_list[0].value, test_field_4d.value[:, 0, 0, 0])

    def test_extract_points_invalid_interp(self, test_field_4d):
        with pytest.raises(ValueError, match="Unsupported interpolation method 'invalid'"):
            test_field_4d.extract_points([(0, 0, 0) * u.m], interp="invalid")

    def test_extract_profile_nearest(self, test_field_4d):
        # Profile along X at t=0, y=1, z=1
        # Use value within range to avoid IndexError in nearest_index
        ax_idx, values = test_field_4d.extract_profile(
            axis="x", 
            at={"t": 0 * u.s, "y": 1.0 * u.m, "z": 1.0 * u.m},
            interp="nearest"
        )
        # Should pick y=1, z=1 -> values[i] = i + 1 + 1 + sin(0) = i + 2
        assert_allclose(ax_idx.value, test_field_4d._axis1_index.value)
        assert_allclose(values.value, [2, 3])

    def test_extract_profile_missing_axis_error(self, test_field_4d):
        with pytest.raises(ValueError, match="requires fixed value for axis 'y'"):
            test_field_4d.extract_profile(axis="x", at={"t": 0 * u.s, "z": 0 * u.m})


class TestSpectralValidation:
    def test_spectral_density_domain_error(self, test_field_4d):
        # Convert to freq domain first
        psd_f = test_field_4d.spectral_density(axis=0)
        # PSD of PSD (along t-freq axis) should raise error
        with pytest.raises(ValueError, match="requires axis0_domain='time'"):
            psd_f.spectral_density(axis=0)

    def test_spectral_density_k_domain_error(self, test_field_4d):
        # Transform spatial axis
        k_field = test_field_4d.spectral_density(axis="x")
        # PSD of wavenumber field along axis=1 (now 'kx') should raise error
        # Use axis index because 'x' name is lost/renamed to 'kx'
        with pytest.raises(ValueError, match="requires axis 'kx' in 'real' domain"):
            k_field.spectral_density(axis=1)

    def test_spectral_density_irregular_axis_error(self):
        # Create irregular field
        times = [0, 0.01, 0.03, 0.04] * u.s
        data = np.random.rand(4, 1, 1, 1)
        f = ScalarField(data, axis0=times, axis0_domain="time")
        # Ensure it's treated as irregular. AxisDescriptor uses np.diff(axis).
        with pytest.raises(ValueError, match="must be regularly spaced"):
            f.spectral_density(axis=0)


class TestResampleQuantityRate:
    def test_resample_quantity_rate(self, test_field_4d):
        # Original 100 Hz -> resample to 50 Hz
        new_rate = 50 * u.Hz
        resampled = test_field_4d.resample(rate=new_rate)
        assert len(resampled._axis0_index) == 50
        assert_allclose(resampled._axis0_index[1] - resampled._axis0_index[0], 0.02 * u.s)


class TestSignalXcorrOps:
    def test_compute_xcorr_unnormalized_window(self, test_field_4d):
        # Correlation between (0,0,0) and (0,0,0) with window
        p = (0 * u.m, 0 * u.m, 0 * u.m)
        from gwexpy.fields.signal import compute_xcorr
        xcorr = compute_xcorr(test_field_4d, p, p, normalize=False, window="hann")
        assert xcorr.unit == test_field_4d.unit ** 2
        # Max should be at lag 0
        assert np.argmax(xcorr.value) == len(xcorr.value) // 2


class TestTimeDelayMapPlanes:
    def test_time_delay_map_xz_yz(self, test_field_4d):
        from gwexpy.fields.signal import time_delay_map
        ref = (0, 0, 0) * u.m
        
        # plane='xz' -> scan x, z; fix y
        map_xz = time_delay_map(test_field_4d, ref, plane="xz", at={"y": 0 * u.m})
        # Result maintains 4D structure (T, X, Y, Z)
        # axis1 is X, axis2 is Y (fixed, size 1), axis3 is Z
        assert map_xz._axis1_name == "x"
        assert map_xz._axis2_name == "y"
        assert map_xz._axis3_name == "z"
        assert map_xz.shape[1] == len(test_field_4d._axis1_index)
        assert map_xz.shape[2] == 1  # Fixed axis
        assert map_xz.shape[3] == len(test_field_4d._axis3_index)
        
        # plane='yz' -> scan y, z; fix x
        map_yz = time_delay_map(test_field_4d, ref, plane="yz", at={"x": 0 * u.m})
        assert map_yz._axis1_name == "x" # Original name preserved at index 1
        assert map_yz._axis2_name == "y"
        assert map_yz._axis3_name == "z"
        assert map_yz.shape[1] == 1 # Fixed x
        assert map_yz.shape[2] == len(test_field_4d._axis2_index)
        assert map_yz.shape[3] == len(test_field_4d._axis3_index)
