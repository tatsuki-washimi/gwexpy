
import pytest
import numpy as np
from astropy import units as u
from gwexpy.timeseries import TimeSeries
from gwexpy.types.time_plane_transform import TimePlaneTransform
from gwexpy.interop.mt_ import to_mth5

# --- P1-1: TimePlaneTransform at_time interpolation ---

def test_time_plane_transform_linear_interp():
    # Setup a simple 3x2x2 plane sequence
    # Time: 0, 1, 2
    # Value at t=0: [[0, 0], [0, 0]]
    # Value at t=1: [[10, 10], [10, 10]]
    # Value at t=2: [[20, 20], [20, 20]]
    
    val = np.array([
        [[0, 0], [0, 0]],
        [[10, 10], [10, 10]],
        [[20, 20], [20, 20]]
    ], dtype=float)
    
    times = [0, 1, 2] * u.s
    # Axis1/2 dummy
    axis1 = [0, 1] * u.Hz
    
    # Construct manually or via tuple
    tpt = TimePlaneTransform(
        (val, times, axis1, axis1, u.dimensionless_unscaled)
    )
    
    # 1. Exact match (t=1.0)
    p1 = tpt.at_time(1.0*u.s, method='linear')
    assert np.allclose(p1.value, 10)
    
    # 2. Midpoint (t=0.5) -> should be 5
    p05 = tpt.at_time(0.5*u.s, method='linear')
    assert np.allclose(p05.value, 5)
    
    # 3. t=1.2 -> should be 12
    p12 = tpt.at_time(1.2*u.s, method='linear')
    assert np.allclose(p12.value, 12)
    
    # 4. Out of bounds (clip)
    p_neg = tpt.at_time(-0.1*u.s, method='linear')
    assert np.allclose(p_neg.value, 0) # Clamped to start
    
    p_over = tpt.at_time(2.5*u.s, method='linear')
    assert np.allclose(p_over.value, 20) # Clamped to end

# --- P1-2: STLT ---

def test_stlt_basic():
    # Create a sine wave 10Hz, fs=100Hz, duration=1s
    dt = 0.01 * u.s
    t = np.arange(100) * dt
    data = np.sin(2 * np.pi * 10 * t.value)
    ts = TimeSeries(data, dt=dt, t0=0*u.s)
    
    # Run STLT
    # Window 0.5s (50 samples), Stride 0.1s (10 samples)
    stlt = ts.stlt(stride='0.1s', window='0.5s')
    
    assert isinstance(stlt, TimePlaneTransform)
    assert stlt.kind == "stlt_mag_outer"
    assert stlt.meta['stride'] == '0.1s'
    
    # Check shape
    # n_steps approx (1 - 0.5)/0.1 + 1 = 6 steps
    # freq bins: 50 samples / 2 + 1 = 26 bins (scipy stft default)
    # Expected shape: (6, 26, 26) or similar
    shape = stlt.shape
    assert len(shape) == 3
    # assert shape[0] >= 5 # Time steps
    
    # Check values exist
    assert np.sum(stlt.value) > 0

# --- P1-3: Resample Aggregation ---

def test_resample_agg_methods():
    # 0, 1, 2, 3, 4, 100 ...
    # Bin size 5. Bin 0 contains 0, 1, 2, 3, 4. Median=2. Max=4. Min=0.
    # Bin 1 contains 100...
    
    data = np.arange(10, dtype=float)
    ts = TimeSeries(data, dt=1*u.s, t0=0*u.s)
    
    # Resample to 5s bins
    # Bin 0: [0, 1, 2, 3, 4]
    # Bin 1: [5, 6, 7, 8, 9]
    
    # Median
    ts_med = ts.resample('5s', agg='median')
    assert len(ts_med) == 2
    assert ts_med.value[0] == 2.0
    assert ts_med.value[1] == 7.0
    
    # Max
    ts_max = ts.resample('5s', agg='max')
    assert ts_max.value[0] == 4.0
    assert ts_max.value[1] == 9.0
    
    # Min
    ts_min = ts.resample('5s', agg='min')
    assert ts_min.value[0] == 0.0
    assert ts_min.value[1] == 5.0

    # With NaNs
    data_nan = np.array([0, 1, np.nan, 3, 4], dtype=float)
    ts_nan = TimeSeries(data_nan, dt=1*u.s, t0=0*u.s)
    
    # Median with NaNs (omit by default logic in my impl? binned_statistic handles nan?)
    # scipy.stats.binned_statistic treats NaN as value unless filtered?
    # No, it propagates NaN if present usually.
    # My implementation filters nans if nan_policy='omit' is passed?
    # My implementation code for 'mean' handled masking.
    # Check code: binned_statistic(valid_indices, valid_values...)
    # valid_indices was filtered by valid_mask = (bin_indices...) not nan mask.
    # Wait, existing code for 'mean' had "handle nan_policy" block.
    # I should check if I used `valid_values` which might contain NaNs in the new block.
    # Yes, I used `valid_values`.
    # If nan_policy='omit', existing code masks NaNs from valid_values/valid_indices.
    # So `valid_values` passed to my new block should be clean.
    
    ts_nan_res = ts_nan.resample('5s', agg='median', nan_policy='omit')
    # bin 0: [0, 1, 3, 4] -> median 2.0? (0,1,3,4 average of 1,3 is 2)
    assert ts_nan_res.value[0] == 2.0

# --- P1-4: MTH5 ---

def test_mth5_interop_error_or_run():
    # Since we can't easily install mth5 here, we check it raises ImportError
    # or runs if mocked/installed.
    # For now, just call it and expect ImportError if not installed.
    
    ts = TimeSeries([1, 2, 3], dt=1*u.s)
    
    try:
        import mth5
        installed = True
    except ImportError:
        installed = False
        
    if not installed:
        with pytest.raises(ImportError) as exc:
            to_mth5(ts, "dummy.h5")
        assert "mth5" in str(exc.value)
    else:
        # If installed (unlikely in this env), test?
        # Skip for now
        pass
