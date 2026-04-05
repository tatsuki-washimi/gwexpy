import numpy as np

from gwexpy.analysis.coupling import CouplingFunctionAnalysis, RatioThreshold
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def test_compare_legacy():
    # Simple setup where SegmentTable vs single PSD should match
    fs = 4096.0
    duration = 10.0
    t = np.arange(0, duration, 1/fs)
    rng = np.random.default_rng(123)

    # Witness
    wit_inj = TimeSeries(rng.normal(0, 10, len(t)), sample_rate=fs, name="W", t0=0)
    wit_bkg = TimeSeries(rng.normal(0, 1, len(t)), sample_rate=fs, name="W", t0=0)

    # Target (CF = 2.0)
    tgt_bkg = TimeSeries(rng.normal(0, 1, len(t)), sample_rate=fs, name="T", t0=0)
    tgt_inj = tgt_bkg + wit_inj * 2.0

    analysis = CouplingFunctionAnalysis()

    # Run with RatioThreshold (legacy path)
    res_ratio = analysis.compute(
        TimeSeriesDict({"W": wit_inj, "T": tgt_inj}),
        TimeSeriesDict({"W": wit_bkg, "T": tgt_bkg}),
        fftlength=1.0,
        threshold_witness=RatioThreshold(10.0),
        threshold_target=RatioThreshold(2.0),
    )

    # We don't have the "old" code to compare against in the same file,
    # but we can compare against expectation (CF approx 2.0)
    cf_val = res_ratio.cf.value
    valid = ~np.isnan(cf_val)
    print(f"Mean CF (Valid): {np.mean(cf_val[valid]):.3f}")
    assert np.allclose(cf_val[valid], 2.0, rtol=0.4) # Noisy CF

    # Now run with PercentileThreshold (new path) and 100% factor
    # This should yield similar results if background is stable
    from gwexpy.analysis.coupling import PercentileThreshold
    res_perc = analysis.compute(
        TimeSeriesDict({"W": wit_inj, "T": tgt_inj}),
        TimeSeriesDict({"W": wit_bkg, "T": tgt_bkg}),
        fftlength=1.0,
        threshold_witness=RatioThreshold(10.0),
        threshold_target=PercentileThreshold(percentile=50.0, factor=1.0), # Median matching
    )

    cf_val_perc = res_perc.cf.value
    valid_perc = ~np.isnan(cf_val_perc)
    print(f"Mean CF Percentile (Valid): {np.mean(cf_val_perc[valid_perc]):.3f}")

    # They should be close
    overlap_mask = valid & valid_perc
    if np.any(overlap_mask):
        corr = np.corrcoef(cf_val[overlap_mask], cf_val_perc[overlap_mask])[0,1]
        print(f"Correlation: {corr:.4f}")
        assert corr > 0.9

if __name__ == "__main__":
    test_compare_legacy()
