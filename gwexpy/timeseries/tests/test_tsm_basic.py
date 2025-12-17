
import os
import sys
import numpy as np
import warnings
from astropy import units as u

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gwexpy.timeseries import TimeSeriesMatrix
from gwpy.timeseries import TimeSeries

def test_tsm_basic():
    print("Testing TimeSeriesMatrix Construction...")
    # 2 rows, 2 cols, 100 samples
    data = np.random.randn(2, 2, 100)
    dt = 0.01
    
    tsm = TimeSeriesMatrix(data, dt=dt)
    
    assert tsm.shape == (2, 2, 100)
    assert tsm.t0.value == 0
    assert len(tsm.times) == 100
    assert np.isclose(tsm.dt.value, dt)
    assert tsm.sample_rate.value == 100.0
    assert tsm.sample_rate.unit == u.Hz
    
    print("Construction OK.")

def test_exclusivity_and_strict_semantics():
    print("Testing Argument Exclusivity and Strict Semantics...")
    data = np.zeros((1, 1, 10))
    
    # 1. Error on conflicting core args
    try:
        TimeSeriesMatrix(data, dt=1, sample_rate=1)
    except ValueError as e:
        assert "give only one of sample_rate or dt" in str(e)
    else:
        raise AssertionError("Failed to raise ValueError for dt+sample_rate")

    try:
        TimeSeriesMatrix(data, dt=1, t0=0, epoch=0)
    except ValueError as e:
        assert "give only one of epoch or t0" in str(e)
    else:
        raise AssertionError("Failed to raise ValueError for t0+epoch")

    # 2. Strict semantics when times is provided
    times = np.arange(10) * 1.0
    ignored_epoch = 999
    
    # Check xindex duplication warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Pass times AND (arg) epoch. Should warn and ignore.
        # Also pass (kwarg) dx. Should warn and ignore.
        # Also pass (kwarg) xindex. Should warn and ignore (Round 6).
        tsm = TimeSeriesMatrix(data, times=times, epoch=ignored_epoch, dx=0.5, xindex=np.zeros(10)) 
        
        # Check warnings
        assert len(w) > 0
        joined_warnings = "".join([str(warn.message) for warn in w])
        print(f"Warnings caught: {joined_warnings}")
        
        assert "given with times, ignoring" in joined_warnings
        # Ensure xindex was mentioned in the generic warning or logically covered
        # We updated the message to include "xindex"
        
        # Verify dx was ignored (derived from times -> dt=1.0, not 0.5)
        assert np.isclose(tsm.dt.value, 1.0)
        
        # Verify epoch attribute was NOT set from the ignored arg
        current_epoch = getattr(tsm, "epoch", None)
        assert current_epoch != ignored_epoch, f"Epoch should not be stored if ignored. Got {current_epoch}"
        
        # Verify times priority (should be 0..9, not 0..0)
        tsm_times = tsm.times.value if hasattr(tsm.times, 'value') else tsm.times
        assert np.array_equal(tsm_times, times)
    
    print("Exclusivity & Strict Semantics OK.")

def test_epoch_non_duplication():
    print("Testing epoch non-duplication in normal construction...")
    data = np.zeros((1, 1, 10))
    tsm = TimeSeriesMatrix(data, epoch=10.0, sample_rate=1.0)
    assert tsm.t0.value == 10.0
    print("Epoch non-duplication OK.")

def test_sample_rate_setter_ndarray_xindex():
    print("Testing sample_rate setter with ndarray xindex...")
    times = np.arange(10, dtype=float)
    data = np.zeros((1, 1, 10))
    tsm = TimeSeriesMatrix(data, times=times)
    
    assert not isinstance(tsm.xindex, u.Quantity)
    tsm.sample_rate = 2.0 # Hz
    assert np.isclose(tsm.xindex[1].value - tsm.xindex[0].value, 0.5)
    print("sample_rate setter with ndarray OK.")

def test_sample_rate_setter_none():
    print("Testing sample_rate=None behavior...")
    data = np.zeros((1, 1, 10))
    tsm = TimeSeriesMatrix(data, dt=1)
    
    assert tsm.xindex is not None
    tsm.sample_rate = None
    assert tsm.xindex is None
    print("sample_rate=None OK.")

def test_slicing_and_access():
    print("Testing slicing and scalar access...")
    data = np.random.randn(2, 2, 10)
    rows = {"r0": {}, "r1": {}}
    cols = {"c0": {}, "c1": {}}
    tsm = TimeSeriesMatrix(data, dt=1, rows=rows, cols=cols)
    
    ts_key = tsm["r0", "c1"]
    assert isinstance(ts_key, TimeSeries)
    assert len(ts_key) == 10
    
    # Check no-copy logical correctness (values match)
    ts_times = ts_key.times.value if hasattr(ts_key.times, 'value') else ts_key.times
    tsm_times = tsm.times.value if hasattr(tsm.times, 'value') else tsm.times
    assert np.array_equal(ts_times, tsm_times)
    
    print("Slicing OK.")

def test_spectral():
    print("Testing spectral methods...")
    data = np.random.randn(2, 2, 100)
    tsm = TimeSeriesMatrix(data, sample_rate=100)
    fsm = tsm.fft()
    assert fsm.shape == (2, 2, 51)
    print("Spectral OK.")

if __name__ == "__main__":
    test_tsm_basic()
    test_exclusivity_and_strict_semantics()
    test_epoch_non_duplication()
    test_sample_rate_setter_ndarray_xindex()
    test_sample_rate_setter_none()
    test_slicing_and_access()
    test_spectral()
    print("ALL TESTS PASSED")
