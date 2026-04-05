import warnings

import numpy as np
import pytest

from gwexpy.analysis.coupling import (
    CouplingFunctionAnalysis,
    PercentileThreshold,
    RatioThreshold,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


@pytest.fixture
def sample_data():
    fs = 100.0
    duration = 20.0
    t = np.arange(0, duration, 1/fs)
    rng = np.random.default_rng(42)

    # Witness
    wit_inj = TimeSeries(rng.normal(0, 10, len(t)), sample_rate=fs, name="WIT", t0=0)
    wit_bkg = TimeSeries(rng.normal(0, 1, len(t)), sample_rate=fs, name="WIT", t0=0)

    # Target
    tgt_inj = TimeSeries(rng.normal(0, 10, len(t)), sample_rate=fs, name="TGT", t0=0)
    tgt_bkg = TimeSeries(rng.normal(0, 1, len(t)), sample_rate=fs, name="TGT", t0=0)

    data_inj = TimeSeriesDict({"WIT": wit_inj, "TGT": tgt_inj})
    data_bkg = TimeSeriesDict({"WIT": wit_bkg, "TGT": tgt_bkg})
    return data_inj, data_bkg, fs

def test_hardening_parallel(sample_data):
    pytest.importorskip("joblib")
    data_inj, data_bkg, fs = sample_data
    analysis = CouplingFunctionAnalysis()

    res = analysis.compute(
        data_inj,
        data_bkg,
        fftlength=1.0,
        threshold_witness=RatioThreshold(2.0),
        threshold_target=PercentileThreshold(percentile=50.0, factor=1.0),
        n_jobs=2,
    )

    from gwexpy.analysis.coupling import CouplingResult
    assert isinstance(res, CouplingResult)
    assert res.target_name == "TGT"

def test_memory_limit_stride_increase(sample_data):
    data_inj, data_bkg, fs = sample_data
    analysis = CouplingFunctionAnalysis()

    # Set a very small memory limit (e.g., 10 KB) to trigger stride adjustment
    # 10s duration, fft=1.0, stride=1.0 -> 10 rows.
    # fs=100 -> 51 freqs.
    # 10 * 51 * 8 * 1.2 overhead approx 5 KB.
    # We set it even smaller to force increase.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = analysis.compute(
            data_inj, data_bkg, fftlength=1.0,
            threshold_target=PercentileThreshold(),
            memory_limit=1000, # 1 KB
            bkg_stride=1.0
        )
        # Verify result is a successful CouplingResult
        from gwexpy.analysis.coupling import CouplingResult
        assert isinstance(res, CouplingResult)
        assert res.target_name == "TGT"

def test_frequency_alignment_interpolate(sample_data):
    data_inj, data_bkg, fs = sample_data

    from gwpy.segments import Segment

    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.table.segment_table import SegmentTable

    psd_tgt_inj = data_inj["TGT"].psd(fftlength=1.0)
    psd_tgt_bkg = data_bkg["TGT"].psd(fftlength=1.0)
    ref_freqs = psd_tgt_inj.frequencies.value

    shifted_small = FrequencySeries(
        np.concatenate([[psd_tgt_bkg.value[0]], psd_tgt_bkg.value]),
        frequencies=np.concatenate([[ref_freqs[0] - 0.25], ref_freqs + 0.25]),
        unit=psd_tgt_bkg.unit,
    )
    shifted_large = FrequencySeries(
        np.concatenate([[psd_tgt_bkg.value[0]], psd_tgt_bkg.value]),
        frequencies=np.concatenate([[ref_freqs[0] - 1.5], ref_freqs + 1.5]),
        unit=psd_tgt_bkg.unit,
    )

    st_small = SegmentTable.from_segments([Segment(0, 1)], psd=[shifted_small])
    st_large = SegmentTable.from_segments([Segment(0, 1)], psd=[shifted_large])

    th_clip = PercentileThreshold(percentile=50, factor=1.0, freq_align="clip")
    with pytest.raises(ValueError, match="No compatible background PSD rows"):
        th_clip.threshold(psd_tgt_inj, psd_tgt_bkg, bkg_table=st_small)

    th_interp = PercentileThreshold(percentile=50, factor=1.0, freq_align="interpolate")
    assert th_interp.threshold(psd_tgt_inj, psd_tgt_bkg, bkg_table=st_small).shape == psd_tgt_inj.value.shape

    with pytest.raises(ValueError, match="No compatible background PSD rows"):
        th_interp.threshold(psd_tgt_inj, psd_tgt_bkg, bkg_table=st_large)


def test_bkg_segment_table_skips_short_boundary_rows(sample_data):
    data_inj, data_bkg, fs = sample_data

    data_bkg["TGT"].t0 = 0.1

    from gwexpy.analysis.coupling import _build_bkg_segment_table

    st = _build_bkg_segment_table(
        data_bkg["TGT"], fftlength=1.0, overlap=0.0, stride=1.0
    )

    assert len(st) > 0

def test_auto_calibrate(sample_data):
    data_inj, data_bkg, fs = sample_data
    analysis = CouplingFunctionAnalysis()

    # Auto-calibrate against background
    cal = analysis.auto_calibrate_percentile_factor(
        data_bkg["WIT"], data_bkg["TGT"], fftlength=1.0
    )

    assert "percentile_factor" in cal
    assert cal["percentile_factor"] > 0
    assert cal["success"] is True

def test_response_hardening_parallel(sample_data):
    pytest.importorskip("joblib")
    data_inj, data_bkg, fs = sample_data
    from gwexpy.analysis.response import ResponseFunctionAnalysis

    # Create simple stepped sine witness (just mock)
    wit = data_inj["WIT"]
    tgt = data_inj["TGT"]

    analysis = ResponseFunctionAnalysis()
    # Manual segments to avoid auto-detect issues with white noise
    segments = [(1.0, 4.0, 10.0), (5.0, 8.0, 20.0)]

    res = analysis.compute(
        wit, tgt, segments=segments, fftlength=1.0, n_jobs=2
    )
    assert len(res.table) == 2
