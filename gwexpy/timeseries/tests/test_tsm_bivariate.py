import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.frequencyseries import FrequencySeriesMatrix


def _build_matrices():
    n = 1024
    dt = 1 / 1024
    t0 = 123.0
    data_a = np.random.randn(2, 2, n)
    data_b = np.random.randn(2, 2, n)
    rows = {"r0": {}, "r1": {}}
    cols = {"c0": {}, "c1": {}}
    a = TimeSeriesMatrix(data_a, dt=dt, t0=t0, rows=rows, cols=cols)
    b = TimeSeriesMatrix(data_b, dt=dt, t0=t0, rows=rows, cols=cols)
    return a, b


def _check_common_freq_axis(fs_matrix):
    if hasattr(fs_matrix, "df") and fs_matrix.df is not None:
        assert fs_matrix.df != 0
    freqs = getattr(fs_matrix, "frequencies", None)
    assert freqs is not None
    assert fs_matrix.shape[2] == len(freqs)


def test_csd_matrix_matrix():
    if not hasattr(GWpyTimeSeries, "csd"):
        print("SKIP: csd not available")
        return
    a, b = _build_matrices()
    out = a.csd(b)
    assert isinstance(out, FrequencySeriesMatrix)
    assert out.shape[:2] == a.shape[:2]
    _check_common_freq_axis(out)
    assert out.epoch == a[0, 0].epoch


def test_csd_broadcast_timeseries():
    if not hasattr(GWpyTimeSeries, "csd"):
        print("SKIP: csd not available")
        return
    a, b = _build_matrices()
    other = b[0, 0]
    out = a.csd(other)
    assert isinstance(out, FrequencySeriesMatrix)
    assert out.shape[:2] == a.shape[:2]


def test_csd_mismatch_errors():
    if not hasattr(GWpyTimeSeries, "csd"):
        print("SKIP: csd not available")
        return
    a, b = _build_matrices()
    try:
        a.csd(b[:1, :])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on shape mismatch")

    try:
        a.csd(1)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError on invalid other type")


def test_transfer_function_sample_rate_mismatch():
    if not hasattr(GWpyTimeSeries, "transfer_function"):
        print("SKIP: transfer_function not available")
        return

    n_low = 4096
    n_high = 8192
    t0 = 123.0
    rows = {"r0": {}, "r1": {}}
    cols = {"c0": {}, "c1": {}}
    a = TimeSeriesMatrix(
        np.random.randn(2, 2, n_low),
        dt=1 / 1024,
        t0=t0,
        rows=rows,
        cols=cols,
    )
    b = TimeSeriesMatrix(
        np.random.randn(2, 2, n_high),
        dt=1 / 2048,
        t0=t0,
        rows=rows,
        cols=cols,
    )
    out = a.transfer_function(b)
    assert isinstance(out, FrequencySeriesMatrix)
    assert out.shape[:2] == a.shape[:2]
    assert out.epoch == a[0, 0].epoch


def test_coherence_and_transfer():
    a, b = _build_matrices()
    if hasattr(GWpyTimeSeries, "coherence"):
        out = a.coherence(b)
        assert isinstance(out, FrequencySeriesMatrix)
        assert out.shape[:2] == a.shape[:2]
        assert np.isrealobj(out.value)
        assert out.epoch == a[0, 0].epoch
    else:
        print("SKIP: coherence not available")

    if hasattr(GWpyTimeSeries, "transfer_function"):
        out_tf = a.transfer_function(b)
        assert isinstance(out_tf, FrequencySeriesMatrix)
        assert out_tf.shape[:2] == a.shape[:2]
        assert out_tf.epoch == a[0, 0].epoch
    else:
        print("SKIP: transfer_function not available")


if __name__ == "__main__":
    test_csd_matrix_matrix()
    test_csd_broadcast_timeseries()
    test_csd_mismatch_errors()
    test_transfer_function_sample_rate_mismatch()
    test_coherence_and_transfer()
    print("ALL BIVARIATE TESTS PASSED")
