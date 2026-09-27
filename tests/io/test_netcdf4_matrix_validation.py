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


@pytest.mark.parametrize("reader", [TimeSeries, TimeSeriesDict, TimeSeriesMatrix])
@pytest.mark.parametrize(
    "times",
    [
        np.array([0, 1, 2, 4, 5], dtype=float),
        np.array([0, 1, 2, 4, 5], dtype="timedelta64[s]") + np.datetime64("2020-01-01"),
    ],
)
def test_public_legacy_read_rejects_irregular_time(tmp_path, reader, times):
    path = tmp_path / "irregular.nc"
    xr.Dataset(
        {"signal": ("time", [10, 11, 12, 14, 15])}, coords={"time": times}
    ).to_netcdf(path)
    with pytest.raises(ValueError, match="irregular.*time"):
        reader.read(path, format="nc")


def test_public_matrix_roundtrip_preserves_unit(tmp_path):
    path = _matrix_file(tmp_path)
    with xr.open_dataset(path) as ds:
        assert {da.attrs["units"] for da in ds.data_vars.values()} == {"V"}
    loaded = TimeSeriesMatrix.read(path, format="nc")
    assert {str(unit) for unit in loaded.units.flat} == {"V"}


def test_public_matrix_read_legacy_single_channel_preserves_unit(tmp_path):
    path = tmp_path / "legacy.nc"
    xr.Dataset(
        {"signal": xr.DataArray([1, 2, 3], dims=["time"], attrs={"units": "V"})},
        coords={"time": [0.0, 1.0, 2.0]},
    ).to_netcdf(path)
    with pytest.warns(RuntimeWarning, match="legacy"):
        loaded = TimeSeriesMatrix.read(path, format="nc")
    assert str(loaded[0, 0].unit) == "V"


def test_public_matrix_write_rejects_mixed_units_before_target_exists(tmp_path):
    path = tmp_path / "mixed.nc"
    source = TimeSeriesMatrix(
        np.ones((2, 2, 3)), t0=0, dt=1, unit=[["V", "V"], ["V", "m"]]
    )
    with pytest.raises(ValueError, match="mixed units"):
        source.write(path, format="nc")
    assert not path.exists()


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


def test_public_read_rejects_one_ulp_numeric_gap_at_gps_epoch(tmp_path):
    path = tmp_path / "one-ulp-gap.nc"
    epoch = 1e9
    times = np.array([epoch, epoch + 1, epoch + 2 + np.spacing(epoch), epoch + 3])
    xr.Dataset({"signal": ("time", [1, 2, 3, 4])}, coords={"time": times}).to_netcdf(
        path
    )
    with pytest.raises(ValueError, match="irregular.*time"):
        TimeSeries.read(path, format="nc")


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


def test_public_read_accepts_regular_numeric_axis_with_overflowing_endpoints(tmp_path):
    path = tmp_path / "large-regular.nc"
    times = (
        np.longdouble(-9e307) + np.arange(4, dtype=np.longdouble) * np.longdouble(6e307)
    ).astype(np.float64)
    xr.Dataset(
        {"signal": ("time", [1.0, 2.0, 3.0, 4.0])}, coords={"time": times}
    ).to_netcdf(path)
    loaded = TimeSeries.read(path, format="nc")
    np.testing.assert_array_equal(loaded.value, [1.0, 2.0, 3.0, 4.0])
    assert np.isclose(loaded.dt.value, 6e307)


def test_public_read_accepts_regular_high_epoch_decimal_cadence(tmp_path):
    path = tmp_path / "regular-high-epoch.nc"
    times = 1e9 + np.arange(5) * 0.125
    xr.Dataset({"signal": ("time", np.arange(5))}, coords={"time": times}).to_netcdf(
        path
    )
    loaded = TimeSeries.read(path, format="nc")
    assert loaded.t0.value == 1e9
    assert loaded.dt.value == 0.125


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
