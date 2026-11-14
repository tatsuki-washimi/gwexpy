import numpy as np
import pytest
from astropy import units as u

from gwexpy.analysis.coupling import (
    CouplingFunctionAnalysis,
    PercentileThreshold,
    RatioThreshold,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def test_compute_with_percentile_threshold():
    # Setup data
    fs = 100.0
    duration = 5.0
    t = np.arange(0, duration, 1 / fs)

    # Witness (High injection, low background)
    wit_inj = TimeSeries(np.random.normal(0, 10, len(t)), sample_rate=fs, name="WIT")
    wit_bkg = TimeSeries(np.random.normal(0, 1, len(t)), sample_rate=fs, name="WIT")

    # Target (High injection, low background)
    tgt_inj = TimeSeries(np.random.normal(0, 10, len(t)), sample_rate=fs, name="TGT")
    tgt_bkg = TimeSeries(np.random.normal(0, 1, len(t)), sample_rate=fs, name="TGT")

    data_inj = TimeSeriesDict({"WIT": wit_inj, "TGT": tgt_inj})
    data_bkg = TimeSeriesDict({"WIT": wit_bkg, "TGT": tgt_bkg})

    analysis = CouplingFunctionAnalysis()
    # Use PercentileThreshold for witness
    # This should trigger _build_bkg_segment_table internally
    res = analysis.compute(
        data_inj,
        data_bkg,
        fftlength=1.0,
        threshold_witness=PercentileThreshold(percentile=50, factor=1.0),
        n_jobs=1,
    )

    # Should return a single CouplingResult if only one target
    from gwexpy.analysis.coupling import CouplingResult

    assert isinstance(res, CouplingResult)
    assert res.target_name == "TGT"
    assert len(res.cf.value) > 0


def test_compute_parallel():
    # Ensure st.materialize() works for Parallel joblib calls
    pytest.importorskip("joblib")
    fs = 100.0
    duration = 5.0
    t = np.arange(0, duration, 1 / fs)

    data_inj = TimeSeriesDict(
        {
            "WIT": TimeSeries(
                np.random.normal(0, 10, len(t)), sample_rate=fs, name="WIT"
            ),
            "T1": TimeSeries(
                np.random.normal(0, 10, len(t)), sample_rate=fs, name="T1"
            ),
            "T2": TimeSeries(
                np.random.normal(0, 10, len(t)), sample_rate=fs, name="T2"
            ),
        }
    )
    data_bkg = TimeSeriesDict(
        {
            "WIT": TimeSeries(
                np.random.normal(0, 1, len(t)), sample_rate=fs, name="WIT"
            ),
            "T1": TimeSeries(np.random.normal(0, 1, len(t)), sample_rate=fs, name="T1"),
            "T2": TimeSeries(np.random.normal(0, 1, len(t)), sample_rate=fs, name="T2"),
        }
    )

    analysis = CouplingFunctionAnalysis()
    # Explicitly use PercentileThreshold to trigger the materialize path
    res = analysis.compute(
        data_inj,
        data_bkg,
        fftlength=1.0,
        threshold_witness=PercentileThreshold(),
        n_jobs=2,
    )
    assert isinstance(res, dict)
    assert len(res) == 2
    assert "T1" in res
    assert "T2" in res


def test_compute_passes_bkg_table_to_target_percentile_threshold():
    class RecordingPercentileThreshold(PercentileThreshold):
        def __init__(self):
            super().__init__(percentile=50, factor=1.0)
            self.seen_bkg_tables = []

        def threshold(self, psd_inj, psd_bkg, raw_bkg=None, **kwargs):
            self.seen_bkg_tables.append(kwargs.get("bkg_table"))
            return super().threshold(psd_inj, psd_bkg, raw_bkg=raw_bkg, **kwargs)

    fs = 100.0
    duration = 5.0
    t = np.arange(0, duration, 1 / fs)

    rng = np.random.default_rng(123)
    data_inj = TimeSeriesDict(
        {
            "WIT": TimeSeries(rng.normal(0, 10, len(t)), sample_rate=fs, name="WIT"),
            "TGT": TimeSeries(rng.normal(0, 10, len(t)), sample_rate=fs, name="TGT"),
        }
    )
    data_bkg = TimeSeriesDict(
        {
            "WIT": TimeSeries(rng.normal(0, 1, len(t)), sample_rate=fs, name="WIT"),
            "TGT": TimeSeries(rng.normal(0, 1, len(t)), sample_rate=fs, name="TGT"),
        }
    )

    threshold_target = RecordingPercentileThreshold()
    analysis = CouplingFunctionAnalysis()
    analysis.compute(
        data_inj,
        data_bkg,
        fftlength=1.0,
        threshold_witness=RatioThreshold(2.0),
        threshold_target=threshold_target,
        n_jobs=1,
    )

    assert threshold_target.seen_bkg_tables
    assert all(table is not None for table in threshold_target.seen_bkg_tables)


def test_target_threshold_internal_attribute_error_is_not_suppressed():
    class BrokenThreshold(RatioThreshold):
        def check(self, psd_inj, psd_bkg, **kwargs):
            return np.zeros_like(psd_inj.value, dtype=bool)

        def threshold(self, psd_inj, psd_bkg, **kwargs):
            raise AttributeError("broken threshold calculation")

    rng = np.random.default_rng(42)

    def series(scale, name):
        return TimeSeries(rng.normal(size=500) * scale, sample_rate=100, name=name)

    injection = TimeSeriesDict({"WIT": series(10, "WIT"), "TGT": series(1, "TGT")})
    background = TimeSeriesDict({"WIT": series(1, "WIT"), "TGT": series(1, "TGT")})
    with pytest.raises(AttributeError, match="broken threshold calculation"):
        CouplingFunctionAnalysis().compute(
            injection,
            background,
            fftlength=1,
            threshold_witness=RatioThreshold(2),
            threshold_target=BrokenThreshold(2),
            n_jobs=1,
        )
