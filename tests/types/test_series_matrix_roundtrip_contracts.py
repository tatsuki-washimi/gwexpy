"""Round-trip metadata contracts for SeriesMatrix-family containers."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.types import MetaData, MetaDataDict, MetaDataMatrix, SeriesMatrix

MATRIX_CLASSES = (SeriesMatrix, TimeSeriesMatrix, FrequencySeriesMatrix)


class Protocol5Only:
    """Pickle helper that cannot be serialized below protocol 5."""

    def __reduce_ex__(self, protocol):
        if protocol < 5:
            raise TypeError("Protocol5Only requires pickle protocol 5")
        return (Protocol5Only, ())


def _metadata_matrix() -> MetaDataMatrix:
    return MetaDataMatrix(
        [
            [
                MetaData(name="elem_r0c0", unit=u.m, channel="H1:E00"),
                MetaData(name="elem_r0c1", unit=u.s, channel="H1:E01"),
            ],
            [
                MetaData(name="elem_r1c0", unit=u.Hz, channel="L1:E10"),
                MetaData(name="elem_r1c1", unit=u.kg, channel="L1:E11"),
            ],
        ]
    )


def _rows() -> MetaDataDict:
    return MetaDataDict(
        {
            "rowA": MetaData(name="row_a", unit=u.m, channel="H1:ROW_A"),
            "rowB": MetaData(name="row_b", unit=u.s, channel="L1:ROW_B"),
        },
        expected_size=2,
        key_prefix="row",
    )


def _cols() -> MetaDataDict:
    return MetaDataDict(
        {
            "colX": MetaData(name="col_x", unit=u.Hz, channel="H1:COL_X"),
            "colY": MetaData(name="col_y", unit=u.kg, channel="L1:COL_Y"),
        },
        expected_size=2,
        key_prefix="col",
    )


def _make_matrix(matrix_cls: type[SeriesMatrix]) -> SeriesMatrix:
    data = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5)
    kwargs = {
        "meta": _metadata_matrix(),
        "rows": _rows(),
        "cols": _cols(),
        "name": "roundtrip-contract",
        "epoch": 1234.5,
        "attrs": {"pipeline": "wave1", "version": 271},
    }
    if matrix_cls is TimeSeriesMatrix:
        return matrix_cls(data, times=np.arange(5) * u.s, **kwargs)
    if matrix_cls is FrequencySeriesMatrix:
        return matrix_cls(data, frequencies=np.arange(5) * u.Hz, **kwargs)
    return matrix_cls(data, xindex=np.arange(5) * u.s, **kwargs)


def _assert_metadata_contract(restored: SeriesMatrix, expected: SeriesMatrix) -> None:
    np.testing.assert_array_equal(restored.view(np.ndarray), expected.view(np.ndarray))
    assert restored.shape == expected.shape

    np.testing.assert_array_equal(
        u.Quantity(restored.xindex).value,
        u.Quantity(expected.xindex).value,
    )
    assert u.Quantity(restored.xindex).unit == u.Quantity(expected.xindex).unit

    assert restored.name == expected.name
    assert restored.epoch == expected.epoch
    assert restored.attrs == expected.attrs

    assert list(restored.rows.keys()) == list(expected.rows.keys())
    assert list(restored.cols.keys()) == list(expected.cols.keys())
    assert restored.rows["rowA"].name == expected.rows["rowA"].name
    assert restored.rows["rowA"].unit == expected.rows["rowA"].unit
    assert str(restored.rows["rowA"].channel) == str(expected.rows["rowA"].channel)
    assert restored.cols["colY"].name == expected.cols["colY"].name
    assert restored.cols["colY"].unit == expected.cols["colY"].unit
    assert str(restored.cols["colY"].channel) == str(expected.cols["colY"].channel)

    assert restored.meta.shape == expected.meta.shape
    assert restored.meta[0, 0].name == expected.meta[0, 0].name
    assert restored.meta[0, 0].unit == expected.meta[0, 0].unit
    assert str(restored.meta[0, 0].channel) == str(expected.meta[0, 0].channel)
    assert restored.meta[1, 1].name == expected.meta[1, 1].name
    assert restored.meta[1, 1].unit == expected.meta[1, 1].unit
    assert str(restored.meta[1, 1].channel) == str(expected.meta[1, 1].channel)


def _xindex_values(matrix: SeriesMatrix) -> np.ndarray:
    return np.asarray(u.Quantity(matrix.xindex).value)


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_copy_preserves_metadata_with_independent_metadata_objects(matrix_cls):
    matrix = _make_matrix(matrix_cls)

    copied = matrix.copy()

    _assert_metadata_contract(copied, matrix)
    copied.view(np.ndarray)[0, 0, 0] = -1
    copied.meta[0, 0].name = "changed"
    copied.rows["rowA"].name = "changed-row"
    copied.attrs["pipeline"] = "changed"

    assert matrix.view(np.ndarray)[0, 0, 0] != -1
    assert matrix.meta[0, 0].name == "elem_r0c0"
    assert matrix.rows["rowA"].name == "row_a"
    assert matrix.attrs["pipeline"] == "wave1"


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_typed_view_preserves_metadata_by_reference(matrix_cls):
    matrix = _make_matrix(matrix_cls)

    viewed = matrix.view(matrix_cls)

    _assert_metadata_contract(viewed, matrix)
    assert viewed.meta is matrix.meta
    assert viewed.rows is matrix.rows
    assert viewed.cols is matrix.cols
    assert viewed.attrs is matrix.attrs


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_pickle_roundtrip_preserves_matrix_metadata_contract(matrix_cls):
    matrix = _make_matrix(matrix_cls)

    restored = pickle.loads(pickle.dumps(matrix))

    assert isinstance(restored, matrix_cls)
    _assert_metadata_contract(restored, matrix)


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_pickle_roundtrip_filters_non_picklable_attrs_entries(matrix_cls):
    matrix = _make_matrix(matrix_cls)

    def callback() -> None:
        return None

    matrix.attrs["runtime_callback"] = callback
    matrix.attrs["nested"] = {"labels": ["H1", "L1"], "active": True}
    expected = matrix.copy()
    del expected.attrs["runtime_callback"]

    with pytest.warns(UserWarning, match="runtime_callback"):
        restored = pickle.loads(pickle.dumps(matrix))

    assert matrix.attrs["runtime_callback"] is callback
    assert isinstance(restored, matrix_cls)
    _assert_metadata_contract(restored, expected)
    assert "runtime_callback" not in restored.attrs
    assert restored.attrs["nested"] == {"labels": ["H1", "L1"], "active": True}


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_default_pickle_filters_attrs_unsafe_for_default_protocol(matrix_cls):
    matrix = _make_matrix(matrix_cls)
    matrix.attrs["default_protocol_unsafe"] = Protocol5Only()
    matrix.attrs["default_protocol_safe"] = "kept"

    with pytest.warns(UserWarning, match="default_protocol_unsafe"):
        restored = pickle.loads(pickle.dumps(matrix))

    assert "default_protocol_unsafe" in matrix.attrs
    assert "default_protocol_unsafe" not in restored.attrs
    assert restored.attrs["default_protocol_safe"] == "kept"


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_pickle_warns_once_listing_all_dropped_attrs_keys(matrix_cls):
    matrix = _make_matrix(matrix_cls)

    def callback() -> None:
        return None

    matrix.attrs["bad_one"] = callback
    matrix.attrs["bad_two"] = Protocol5Only()
    matrix.attrs["good"] = "kept"

    with pytest.warns(UserWarning) as record:
        restored = pickle.loads(pickle.dumps(matrix))

    dropped_warnings = [
        w for w in record if "cannot be pickled" in str(w.message)
    ]
    assert len(dropped_warnings) == 1
    message = str(dropped_warnings[0].message)
    assert "bad_one" in message
    assert "bad_two" in message
    assert "bad_one" not in restored.attrs
    assert "bad_two" not in restored.attrs
    assert restored.attrs["good"] == "kept"
    assert restored.attrs["pipeline"] == "wave1"


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_pickle_filters_attrs_with_custom_reduce_raising(matrix_cls):
    matrix = _make_matrix(matrix_cls)

    class BadReduce:
        def __reduce__(self):
            raise ValueError("cannot pickle this")

    matrix.attrs["bad_reduce"] = BadReduce()
    matrix.attrs["good"] = "kept"

    with pytest.warns(UserWarning, match="bad_reduce"):
        restored = pickle.loads(pickle.dumps(matrix))

    assert "bad_reduce" not in restored.attrs
    assert restored.attrs["good"] == "kept"


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_setstate_backward_compat_without_metadata_dict(matrix_cls):
    matrix = _make_matrix(matrix_cls)
    reconstruct, args, state = matrix.__reduce__()
    old_state = state[:-1]

    restored = reconstruct(*args)
    restored.__setstate__(old_state)

    assert isinstance(restored, matrix_cls)
    np.testing.assert_array_equal(restored.view(np.ndarray), matrix.view(np.ndarray))
    assert restored.xindex is None
    assert not hasattr(restored, "meta")
    assert not hasattr(restored, "rows")
    assert not hasattr(restored, "cols")
    assert not hasattr(restored, "name")
    assert not hasattr(restored, "epoch")
    assert not hasattr(restored, "attrs")


def test_pickle_ignores_runtime_gwex_attributes_not_in_metadata_contract():
    matrix = _make_matrix(SeriesMatrix)

    def callback() -> None:
        return None

    matrix._gwex_runtime_callback = callback

    restored = pickle.loads(pickle.dumps(matrix))

    _assert_metadata_contract(restored, matrix)
    assert not hasattr(restored, "_gwex_runtime_callback")


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_hdf5_roundtrip_preserves_matrix_metadata_contract(tmp_path, matrix_cls):
    pytest.importorskip("h5py")
    matrix = _make_matrix(matrix_cls)
    path = tmp_path / f"{matrix_cls.__name__}.h5"

    matrix.write(path, format="hdf5")
    restored = matrix_cls.read(path, format="hdf5")

    assert isinstance(restored, matrix_cls)
    _assert_metadata_contract(restored, matrix)


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
@pytest.mark.parametrize("format_name", ("wide", "long"))
def test_to_pandas_is_a_value_export_not_metadata_roundtrip(matrix_cls, format_name):
    pd = pytest.importorskip("pandas")
    matrix = _make_matrix(matrix_cls)

    df = matrix.to_pandas(format=format_name)
    expected_xindex = _xindex_values(matrix)
    expected_columns = [
        "rowA_colX",
        "rowA_colY",
        "rowB_colX",
        "rowB_colY",
    ]

    assert isinstance(df, pd.DataFrame)
    if format_name == "wide":
        assert df.shape == (5, 4)
        assert list(df.columns) == expected_columns
        np.testing.assert_array_equal(df.index.to_numpy(), expected_xindex)
        assert not isinstance(df.index.to_numpy(), u.Quantity)
        if isinstance(matrix.xindex, u.Quantity):
            assert df.index.name == f"index [{u.Quantity(matrix.xindex).unit}]"
        np.testing.assert_array_equal(df.to_numpy().T.reshape(2, 2, 5), matrix.value)
    else:
        assert list(df.columns) == ["index", "row", "col", "value"]
        assert len(df) == matrix.size
        assert set(df["row"]) == {"rowA", "rowB"}
        assert set(df["col"]) == {"colX", "colY"}
        expected_long_index = np.tile(expected_xindex, matrix.shape[0] * matrix.shape[1])
        np.testing.assert_array_equal(df["index"].to_numpy(), expected_long_index)
        assert not isinstance(df["index"].to_numpy(), u.Quantity)
        np.testing.assert_array_equal(df["value"].to_numpy(), matrix.value.reshape(-1))

    assert "elem_r0c0" not in df.to_csv(index=True)
    assert "H1:E00" not in df.to_csv(index=True)
    assert "row_a" not in df.to_csv(index=True)
    assert "pipeline" not in df.to_csv(index=True)


@pytest.mark.parametrize("matrix_cls", MATRIX_CLASSES)
def test_csv_write_is_a_wide_value_export_not_metadata_roundtrip(tmp_path, matrix_cls):
    pd = pytest.importorskip("pandas")
    matrix = _make_matrix(matrix_cls)
    path = tmp_path / f"{matrix_cls.__name__}.csv"
    expected_xindex = _xindex_values(matrix)

    matrix.write(path, format="csv")
    df = pd.read_csv(path, index_col=0)

    assert df.shape == (5, 4)
    assert list(df.columns) == ["rowA_colX", "rowA_colY", "rowB_colX", "rowB_colY"]
    np.testing.assert_array_equal(df.index.to_numpy(), expected_xindex)
    assert not isinstance(df.index.to_numpy(), u.Quantity)
    if isinstance(matrix.xindex, u.Quantity):
        assert df.index.name == f"index [{u.Quantity(matrix.xindex).unit}]"
    np.testing.assert_array_equal(df.to_numpy().T.reshape(2, 2, 5), matrix.value)
    csv_text = path.read_text()
    assert "elem_r0c0" not in csv_text
    assert "H1:E00" not in csv_text
    assert "row_a" not in csv_text
    assert "pipeline" not in csv_text
