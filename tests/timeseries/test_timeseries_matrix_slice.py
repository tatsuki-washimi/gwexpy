import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeriesMatrix


def _make_matrix() -> TimeSeriesMatrix:
    data = np.arange(2 * 3 * 10).reshape(2, 3, 10)
    return TimeSeriesMatrix(
        data,
        dt=0.1 * u.s,
        t0=0.0 * u.s,
        rows=["r0", "r1"],
        cols=["c0", "c1", "c2"],
    )


def test_slice_preserves_matrix_type() -> None:
    tsm = _make_matrix()
    sliced = tsm[:, :, 2:5]
    assert type(sliced) is TimeSeriesMatrix
    assert sliced.shape == (2, 3, 3)


def test_index_preserves_matrix_type() -> None:
    tsm = _make_matrix()
    sliced = tsm[0:1, 1:3, :]
    assert type(sliced) is TimeSeriesMatrix
    assert sliced.shape == (1, 2, 10)


def test_crop_preserves_matrix_type() -> None:
    tsm = _make_matrix()
    cropped = tsm.crop(0.2 * u.s, 0.5 * u.s)
    assert type(cropped) is TimeSeriesMatrix
    assert cropped.shape == (2, 3, 3)


def test_label_slice_then_crop_preserves_matrix_type() -> None:
    tsm = _make_matrix()
    sliced = tsm["r0", ["c1", "c2"], :]
    assert type(sliced) is TimeSeriesMatrix
    cropped = sliced.crop(0.1 * u.s, 0.4 * u.s)
    assert type(cropped) is TimeSeriesMatrix
    assert cropped.shape == (1, 2, 3)


# --- Core properties (matrix_core.py) ---

def test_dt_property() -> None:
    tsm = _make_matrix()
    assert tsm.dt.to("s").value == pytest.approx(0.1)


def test_t0_property() -> None:
    tsm = _make_matrix()
    assert tsm.t0 == pytest.approx(0.0 * u.s)


def test_times_property() -> None:
    tsm = _make_matrix()
    times = tsm.times
    assert len(times) == 10
    np.testing.assert_allclose(times.value[:3], [0.0, 0.1, 0.2], atol=1e-10)


def test_span_property() -> None:
    tsm = _make_matrix()
    span = tsm.span
    assert span is not None


def test_sample_rate_property() -> None:
    tsm = _make_matrix()
    sr = tsm.sample_rate
    assert sr.to("Hz").value == pytest.approx(10.0)


def test_sample_rate_setter() -> None:
    tsm = _make_matrix()
    tsm.sample_rate = 20 * u.Hz
    assert tsm.dt.to("s").value == pytest.approx(0.05)
    assert tsm.sample_rate.to("Hz").value == pytest.approx(20.0)


def test_sample_rate_setter_scalar() -> None:
    tsm = _make_matrix()
    tsm.sample_rate = 5  # Hz as plain int
    assert tsm.sample_rate.to("Hz").value == pytest.approx(5.0)


# --- _apply_timeseries_method (via detrend) ---

def test_apply_timeseries_method_detrend() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(2, 1, 100)).astype(float)
    tsm = TimeSeriesMatrix(data, dt=0.01 * u.s, t0=0.0 * u.s)
    result = tsm.detrend()
    assert type(result) is TimeSeriesMatrix
    assert result.shape == tsm.shape
    # After detrending, mean should be near zero
    np.testing.assert_allclose(result.value.mean(axis=-1), 0.0, atol=1e-10)


def test_apply_timeseries_method_not_implemented() -> None:
    tsm = _make_matrix()
    with pytest.raises((NotImplementedError, AttributeError)):
        tsm._apply_timeseries_method("nonexistent_method_xyz")


# --- _apply_timeseries_method via resample ---

def test_apply_timeseries_method_resample() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 1, 200)).astype(float)
    tsm = TimeSeriesMatrix(data, dt=1 / 100 * u.s, t0=0.0 * u.s)
    result = tsm.resample(50)
    assert type(result) is TimeSeriesMatrix
    assert result.shape[0] == 2
    assert result.shape[2] == 100  # half the samples at 50 Hz
