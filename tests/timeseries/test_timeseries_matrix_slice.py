import numpy as np
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
