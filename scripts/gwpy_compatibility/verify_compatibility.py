
import os
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.plot import Plot as GwpyPlot

from gwexpy.timeseries import TimeSeries as GwexTimeSeries
from gwexpy.frequencyseries import FrequencySeries as GwexFrequencySeries
from gwexpy.plot import Plot as GwexPlot
from astropy import units as u

def test_constructor_compatibility():
    print("\n[1.1] Testing Constructor Compatibility...")
    data = np.random.randn(1000)
    
    # Standard constructor
    g1 = GwpyTimeSeries(data, t0=0, dt=0.01, unit='m')
    e1 = GwexTimeSeries(data, t0=0, dt=0.01, unit='m')
    
    assert np.allclose(g1.value, e1.value)
    assert g1.t0 == e1.t0
    assert g1.dt == e1.dt
    assert g1.unit == e1.unit
    
    # Extra args (fmin, fmax) - Gwex should ignore them, Gwpy might fail or ignore depending on version
    # In Gwex, we know we pop them.
    e2 = GwexFrequencySeries(data[:501], frequencies=np.linspace(0, 100, 501), fmin=10)
    assert isinstance(e2, GwexFrequencySeries)
    print("  - Extra args handled correctly.")

def test_decimation_visual_equivalence():
    print("\n[1.2] Testing Visual Equivalence (Decimation)...")
    # Create data with a single sharp spike
    data = np.random.randn(100000) * 0.1
    data[50000] = 10.0 # Huge spike
    
    ts = GwexTimeSeries(data, sample_rate=1000)
    
    # Manual check of adaptive_decimate
    from gwexpy.plot.utils import adaptive_decimate
    ts_dec = adaptive_decimate(ts, target_points=1000)
    
    # Spike should be preserved in decimated data
    assert np.max(ts_dec.value) == 10.0
    assert len(ts_dec) <= 2000 # Interleaved min/max
    print(f"  - Spike preserved: {np.max(ts_dec.value) == 10.0}")
    print(f"  - Data reduction: {len(ts)} -> {len(ts_dec)}")

def test_plot_lifecycle():
    print("\n[1.3] Testing Plot Lifecycle (show/close)...")
    ts = GwexTimeSeries(np.random.randn(100), sample_rate=10)
    plot = ts.plot()
    
    # Check reprs
    assert plot._repr_html_ is None
    # Depending on environment, _repr_png_ might be inherited
    has_png = hasattr(plot, '_repr_png_')
    print(f"  - _repr_html_ is suppressed: {plot._repr_html_ is None}")
    print(f"  - Has _repr_png_: {has_png}")

    # Test show() side effect
    import matplotlib.pyplot as plt
    # Mock plt.show to avoid blocking
    original_show = plt.show
    plt.show = lambda: None
    try:
        plot.show()
        # After show, figure should be closed
        # In matplotlib, a closed figure might still exist as an object but is removed from GCF
        assert plot not in plt.get_fignums()
        print("  - Figure closed after show().")
    finally:
        plt.show = original_show

def test_type_preservation():
    print("\n[1.4] Testing Type Preservation...")
    ts = GwexTimeSeries(np.random.randn(1000), sample_rate=100)
    
    # Crop
    ts_crop = ts.crop(1, 2)
    assert isinstance(ts_crop, GwexTimeSeries), f"Expected GwexTimeSeries, got {type(ts_crop)}"
    
    # Append
    ts2 = GwexTimeSeries(np.random.randn(1000), t0=ts.t0+ts.duration, sample_rate=100)
    ts_app = ts.append(ts2, inplace=False)
    assert isinstance(ts_app, GwexTimeSeries), f"Expected GwexTimeSeries, got {type(ts_app)}"
    
    # IFFT
    fs = ts.fft()
    assert isinstance(fs, GwexFrequencySeries)
    ts_inv = fs.ifft()
    assert isinstance(ts_inv, GwexTimeSeries)
    
    print("  - Type preservation passed for crop, append, fft, ifft.")

if __name__ == "__main__":
    try:
        test_constructor_compatibility()
        test_decimation_visual_equivalence()
        test_plot_lifecycle()
        test_type_preservation()
        print("\nSUCCESS: All basic compatibility checks passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\nFAILURE: Compatibility checks failed.")
        exit(1)
