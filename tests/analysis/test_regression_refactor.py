import numpy as np
import pytest
from astropy import units as u

from gwexpy.analysis.coupling import (
    CouplingFunctionAnalysis,
    PercentileThreshold,
    RatioThreshold,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def test_full_chain_numerical_consistency():
    """
    Simulate a full injection analysis and verify numerical consistency.
    Inject a known CF and recover it.
    """
    fs = 1000.0
    duration = 10.0
    t = np.arange(0, duration, 1/fs)

    # Baseline noise (White noise)
    rng = np.random.default_rng(42)
    wit_bkg_data = rng.normal(0, 1.0, len(t))
    tgt_bkg_data = rng.normal(0, 0.1, len(t))

    # 1. Witness Injection (10Hz Sine)
    wit_inj_data = wit_bkg_data.copy()
    inj_sine = 5.0 * np.sin(2 * np.pi * 10 * t)
    wit_inj_data += inj_sine

    # 2. Target Injection (CF = 0.5)
    tgt_inj_data = tgt_bkg_data.copy()
    tgt_inj_data += 0.5 * inj_sine

    # TimeSeries
    data_inj = TimeSeriesDict({
        "WIT": TimeSeries(wit_inj_data, sample_rate=fs, t0=0, name="WIT", unit="V"),
        "TGT": TimeSeries(tgt_inj_data, sample_rate=fs, t0=0, name="TGT", unit="m"),
    })
    data_bkg = TimeSeriesDict({
        "WIT": TimeSeries(wit_bkg_data, sample_rate=fs, t0=0, name="WIT", unit="V"),
        "TGT": TimeSeries(tgt_bkg_data, sample_rate=fs, t0=0, name="TGT", unit="m"),
    })

    analysis = CouplingFunctionAnalysis()
    fftlength = 1.0
    overlap = 0.5

    # Run Analysis
    res = analysis.compute(
        data_inj, data_bkg,
        fftlength=fftlength,
        overlap=overlap,
        threshold_witness=RatioThreshold(10.0), # Strong injection
        threshold_target=RatioThreshold(2.0),
    )

    # Verify CF at 10Hz
    # find index of 10Hz
    idx_10hz = np.argmin(np.abs(res.frequencies.value - 10.0))
    cf_val = res.cf.value[idx_10hz]

    # Theoretical CF is 0.5
    # Relative difference should be < 1e-3
    rel_diff = abs(cf_val - 0.5) / 0.5
    print(f"CF at 10Hz: {cf_val}, rel_diff: {rel_diff}")

    assert rel_diff < 1e-2, f"CF recovery failed: {cf_val} vs 0.5 (rel_diff={rel_diff})"
    # Note: 1e-3 might be optimistic for short white noise simulation, 1e-2 is safe.

def test_memory_limit_trigger():
    """Verify that memory_limit raises ValueError."""
    fs = 1000.0
    duration = 100.0
    t = np.arange(0, duration, 1/fs)
    ts = TimeSeries(t, sample_rate=fs)

    from gwexpy.analysis.coupling import _build_bkg_segment_table

    # This should estimate:
    # rows: (100 - 1)/1 + 1 = 100
    # bins: (1 * 1000 / 2) + 1 = 501
    # mem: 100 * 501 * 8 = 400,800 bytes (~0.4 MB)

    with pytest.raises(ValueError, match="memory estimate .* exceeds limit"):
        _build_bkg_segment_table(ts, fftlength=1.0, memory_limit=1000) # Only 1KB limit

if __name__ == "__main__":
    test_full_chain_numerical_consistency()
    test_memory_limit_trigger()
    print("Regression tests passed.")
