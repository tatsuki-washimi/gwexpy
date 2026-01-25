import numpy as np

from gwexpy.frequencyseries import FrequencySeriesMatrix


def _make_matrix() -> FrequencySeriesMatrix:
    data = np.arange(2 * 3 * 10).reshape(2, 3, 10)
    return FrequencySeriesMatrix(
        data,
        df=1.0,
        f0=0.0,
        rows=["r0", "r1"],
        cols=["c0", "c1", "c2"],
    )


def test_slice_preserves_matrix_type() -> None:
    fsm = _make_matrix()
    sliced = fsm[:, :, 2:5]
    assert type(sliced) is FrequencySeriesMatrix
    assert sliced.shape == (2, 3, 3)


def test_index_preserves_matrix_type() -> None:
    fsm = _make_matrix()
    sliced = fsm[0:1, 1:3, :]
    assert type(sliced) is FrequencySeriesMatrix
    assert sliced.shape == (1, 2, 10)


def test_crop_preserves_matrix_type() -> None:
    fsm = _make_matrix()
    cropped = fsm.crop(2, 5)
    assert type(cropped) is FrequencySeriesMatrix
    assert cropped.shape == (2, 3, 3)


def test_label_slice_then_crop_preserves_matrix_type() -> None:
    fsm = _make_matrix()
    sliced = fsm["r0", ["c1", "c2"], :]
    assert type(sliced) is FrequencySeriesMatrix
    cropped = sliced.crop(1, 4)
    assert type(cropped) is FrequencySeriesMatrix
    assert cropped.shape == (1, 2, 3)
