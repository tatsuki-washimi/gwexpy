"""Edge-case row/column key round-trip tests across matrix I/O backends (C4).

Hypothesis is not a hard test dependency, so these use an explicit, exhaustive
parametrization of the awkward key shapes instead of property generation:
strings with separators / unicode / empty, integers, floats (incl. NaN/inf),
and nested / mixed tuples.  HDF5 is always exercised; NetCDF4 and Zarr are
exercised when their optional backends are importable.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataDict

# (id, row_key_a, row_key_b) -- two distinct keys of the same awkward shape
KEY_CASES = [
    ("plain_str", "rowA", "rowB"),
    ("slash_colon", "a/b:c", "d/e:f"),
    ("unicode", "unié中", "αβ"),
    ("empty_and_space", "", " "),
    ("ints", 10, 11),
    ("floats", 1.5, 2.5),
    ("tuple2", ("H1", "X"), ("L1", "Y")),
    ("nested_tuple", (("a", "b"), "c"), (("d", "e"), "f")),
    ("mixed_tuple", (1, ("x", 2)), (3, ("y", 4))),
]


def _make_matrix(row_a, row_b):
    data = np.arange(2 * 1 * 4, dtype=float).reshape(2, 1, 4)
    m = TimeSeriesMatrix(data, t0=1_000_000_000.0, sample_rate=8.0)
    m.rows = MetaDataDict(
        OrderedDict({row_a: MetaData(), row_b: MetaData()}),
        expected_size=2,
        key_prefix="row",
    )
    m.cols = MetaDataDict(
        OrderedDict({("col", 0): MetaData()}), expected_size=1, key_prefix="col"
    )
    return m, data


def _assert_keys_equal(got, expected):
    """Compare keys with NaN-awareness (NaN != NaN under ==)."""
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        if isinstance(e, float) and math.isnan(e):
            assert isinstance(g, float) and math.isnan(g)
        else:
            assert g == e, f"{g!r} != {e!r}"


@pytest.mark.parametrize("case", KEY_CASES, ids=[c[0] for c in KEY_CASES])
def test_hdf5_matrix_key_roundtrip(tmp_path, case):
    pytest.importorskip("h5py")
    _id, row_a, row_b = case
    matrix, data = _make_matrix(row_a, row_b)

    path = tmp_path / f"keys_{_id}.h5"
    matrix.write(path, format="hdf5")
    restored = TimeSeriesMatrix.read(path, format="hdf5")

    _assert_keys_equal(list(restored.row_keys()), [row_a, row_b])
    _assert_keys_equal(list(restored.col_keys()), [("col", 0)])
    np.testing.assert_allclose(restored.value, data)


@pytest.mark.parametrize("case", KEY_CASES, ids=[c[0] for c in KEY_CASES])
def test_zarr_matrix_key_roundtrip(tmp_path, case):
    pytest.importorskip("zarr")
    if os.environ.get("GWEXPY_ALLOW_ZARR", "") != "1":
        pytest.skip("zarr tests require GWEXPY_ALLOW_ZARR=1")
    _id, row_a, row_b = case
    matrix, data = _make_matrix(row_a, row_b)

    path = tmp_path / f"keys_{_id}.zarr"
    matrix.write(str(path), format="zarr")
    restored = TimeSeriesMatrix.read(str(path), format="zarr")

    _assert_keys_equal(list(restored.row_keys()), [row_a, row_b])
    np.testing.assert_allclose(restored.value, data)


@pytest.mark.parametrize("case", KEY_CASES, ids=[c[0] for c in KEY_CASES])
def test_netcdf4_matrix_key_roundtrip(tmp_path, case):
    pytest.importorskip("netCDF4")
    pytest.importorskip("xarray")
    _id, row_a, row_b = case
    matrix, data = _make_matrix(row_a, row_b)

    path = tmp_path / f"keys_{_id}.nc"
    matrix.write(str(path), format="nc")
    restored = TimeSeriesMatrix.read(str(path), format="nc")

    _assert_keys_equal(list(restored.row_keys()), [row_a, row_b])
    np.testing.assert_allclose(restored.value, data)
