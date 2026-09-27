"""Public NetCDF reads reject invalid topology and retain physical metadata."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _matrix_file(tmp_path, mutation=None):
    path = tmp_path / "matrix.nc"
    source = TimeSeriesMatrix(
        np.array([[[11, 11, 11], [12, 12, 12]], [[21, 21, 21], [22, 22, 22]]]),
        t0=0,
        dt=1,
        unit="V",
    )
    source.write(path, format="nc")
    with xr.open_dataset(path) as original:
        ds = original.load()
    if mutation is not None:
        mutation(ds)
        ds.to_netcdf(path, mode="w")
    return path


def _by_cell(ds):
    return {
        (da.attrs["gwexpy_row_index"], da.attrs["gwexpy_col_index"]): name
        for name, da in ds.data_vars.items()
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda ds: ds.__delitem__(_by_cell(ds)[1, 1]), "missing"),
        (
            lambda ds: ds.__setitem__(
                "duplicate", ds[_by_cell(ds)[0, 0]].copy(deep=True)
            ),
            "duplicate",
        ),
        (
            lambda ds: ds[_by_cell(ds)[1, 1]].attrs.__setitem__("gwexpy_row_index", 2),
            "row",
        ),
        (
            lambda ds: ds[_by_cell(ds)[1, 1]].attrs.__setitem__("gwexpy_col_index", 2),
            "column",
        ),
        (
            lambda ds: ds[_by_cell(ds)[1, 1]].attrs.__setitem__("gwexpy_row_index", -1),
            "row",
        ),
        (
            lambda ds: ds[_by_cell(ds)[1, 1]].attrs.__setitem__(
                "gwexpy_col_index", 1000
            ),
            "column",
        ),
        (
            lambda ds: ds[_by_cell(ds)[1, 1]].attrs.__setitem__(
                "gwexpy_row_key", '"row0"'
            ),
            "row",
        ),
        (
            lambda ds: ds[_by_cell(ds)[1, 1]].attrs.__setitem__(
                "gwexpy_col_key", '"col0"'
            ),
            "column",
        ),
        (
            lambda ds: ds[_by_cell(ds)[1, 1]].attrs.pop("gwexpy_row_key"),
            "missing",
        ),
        (
            lambda ds: ds[_by_cell(ds)[1, 1]].attrs.__setitem__("units", "m"),
            "unit",
        ),
    ],
)
def test_public_matrix_read_rejects_bad_topology_or_units(tmp_path, mutation, message):
    path = _matrix_file(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        TimeSeriesMatrix.read(path, format="nc")


def test_public_matrix_read_rejects_partially_missing_cell_metadata(tmp_path):
    path = tmp_path / "partial.nc"
    TimeSeriesMatrix(np.ones((1, 1, 3)), t0=0, dt=1).write(path, format="nc")
    with xr.open_dataset(path) as original:
        ds = original.load()
    only = next(iter(ds.data_vars.values()))
    only.attrs.pop("gwexpy_row_key")
    ds.to_netcdf(path, mode="w")
    with pytest.raises(ValueError, match="missing.*row"):
        TimeSeriesMatrix.read(path, format="nc")


@pytest.mark.parametrize("axis", ["row", "col"])
def test_public_matrix_read_rejects_removed_final_axis(tmp_path, axis):
    path = _matrix_file(tmp_path)
    with xr.open_dataset(path) as original:
        ds = original.load()
    for name, da in list(ds.data_vars.items()):
        if da.attrs[f"gwexpy_{axis}_index"] == 1:
            del ds[name]
    ds.to_netcdf(path, mode="w")
    with pytest.raises(ValueError, match="declared.*(row|column)"):
        TimeSeriesMatrix.read(path, format="nc")


def test_public_matrix_read_rejects_declared_matrix_without_cell_metadata(tmp_path):
    path = tmp_path / "unmarked.nc"
    TimeSeriesMatrix(np.ones((1, 1, 3)), t0=0, dt=1).write(path, format="nc")
    with xr.open_dataset(path) as original:
        ds = original.load()
    only = next(iter(ds.data_vars.values()))
    for name in (
        "gwexpy_row_key",
        "gwexpy_col_key",
        "gwexpy_row_index",
        "gwexpy_col_index",
    ):
        only.attrs.pop(name)
    ds.to_netcdf(path, mode="w")
    with pytest.raises(ValueError, match="declared.*cell"):
        TimeSeriesMatrix.read(path, format="nc")


def test_public_matrix_read_accepts_older_v2_without_declared_shape(tmp_path):
    path = _matrix_file(tmp_path)
    with xr.open_dataset(path) as original:
        ds = original.load()
    ds.attrs.pop("gwexpy_matrix_rows")
    ds.attrs.pop("gwexpy_matrix_columns")
    ds.to_netcdf(path, mode="w")
    loaded = TimeSeriesMatrix.read(path, format="nc")
    assert loaded.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("first_values", "reject"),
    [
        (np.array([1, 2, 3], dtype=np.int32), False),
        (np.array([2**53 + 1, 2, 3], dtype=np.int64), True),
    ],
)
def test_public_matrix_read_handles_heterogeneous_numeric_cell_dtypes(
    tmp_path, first_values, reject
):
    path = tmp_path / "mixed-dtypes.nc"
    timing = {
        "gwexpy_netcdf_schema_version": 2,
        "gwexpy_t0_float_hex": (0.0).hex(),
        "gwexpy_t0_gps_seconds": 0,
        "gwexpy_t0_gps_nanoseconds": 0,
        "gwexpy_dt_numerator": "1",
        "gwexpy_dt_denominator": "1",
        "gwexpy_axis_encoding": "t(i)=t0+i*dt",
        "gwexpy_matrix_rows": 1,
        "gwexpy_matrix_columns": 2,
    }

    def attrs(col):
        return {
            "gwexpy_row_key": '"r0"',
            "gwexpy_col_key": f'"c{col}"',
            "gwexpy_key_format": "json",
            "gwexpy_row_index": 0,
            "gwexpy_col_index": col,
            "units": "V",
        }

    xr.Dataset(
        {
            "first": xr.DataArray(first_values, dims=["sample"], attrs=attrs(0)),
            "second": xr.DataArray(
                np.array([1.5, 2.25, 3.75]), dims=["sample"], attrs=attrs(1)
            ),
        },
        coords={"sample": np.arange(3, dtype=np.int64)},
        attrs=timing,
    ).to_netcdf(path)
    if reject:
        with pytest.raises(ValueError, match="no safe common representation"):
            TimeSeriesMatrix.read(path, format="nc")
        return
    loaded = TimeSeriesMatrix.read(path, format="nc")
    np.testing.assert_array_equal(loaded.value[0, 1], [1.5, 2.25, 3.75])


def test_public_matrix_read_rejects_empty_declared_matrix(tmp_path):
    path = _matrix_file(tmp_path)
    with xr.open_dataset(path) as original:
        ds = original.load()
    for name in list(ds.data_vars):
        del ds[name]
    ds.to_netcdf(path, mode="w")
    with pytest.raises(ValueError, match="declared matrix.*no cells"):
        TimeSeriesMatrix.read(path, format="nc")


@pytest.mark.parametrize("axis", ["row", "col"])
@pytest.mark.parametrize("bad_index", [0, 3, 1000])
def test_public_matrix_read_rejects_consistent_axis_index_corruption(
    tmp_path, axis, bad_index
):
    path = _matrix_file(tmp_path)
    with xr.open_dataset(path) as original:
        ds = original.load()
    for da in ds.data_vars.values():
        if da.attrs[f"gwexpy_{axis}_index"] == 1:
            da.attrs[f"gwexpy_{axis}_index"] = bad_index
    ds.to_netcdf(path, mode="w")
    message = "conflicting keys" if bad_index == 0 else "sparse or out of range"
    with pytest.raises(ValueError, match=message):
        TimeSeriesMatrix.read(path, format="nc")
