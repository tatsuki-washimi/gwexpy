"""Focused value and copy-site gates for the X/#586 native readers."""

from __future__ import annotations

import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix


def _copy_site_calls(read, *, function_name: str, filename: str, samples: int):
    """Count only full-cell ``astype`` calls at the candidate source site."""
    calls = 0

    def profile(frame, event, function):
        nonlocal calls
        if event != "c_call" or getattr(function, "__name__", None) != "astype":
            return
        array = getattr(function, "__self__", None)
        if (
            frame.f_code.co_name == function_name
            and frame.f_code.co_filename.endswith(filename)
            and isinstance(array, np.ndarray)
            and array.size == samples
        ):
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = read()
    finally:
        sys.setprofile(previous)
    return result, calls


def _write_ats(path: Path, values: np.ndarray, lsb_mv: float) -> None:
    bits = values.dtype.itemsize * 8
    header = bytearray(1024)
    struct.pack_into(
        "<HhIfId",
        header,
        0,
        1024,
        81 if bits == 64 else 80,
        values.size,
        256.0,
        1_600_000_000,
        lsb_mv,
    )
    header[0x26:0x28] = b"Ex"
    header[0x28:0x2E] = b"MFS06 "
    header[0x84:0x90] = b"ADU08E      "
    struct.pack_into("<h", header, 0xAA, 1 if bits == 64 else 0)
    path.write_bytes(header + values.tobytes())


@pytest.mark.parametrize("dtype", ["<i4", "<i8"])
def test_ats_scaled_bits_metadata_and_no_full_cast(tmp_path, dtype):
    raw = np.array([0, 1, -1, 2**20 + 1, -(2**20 + 1)], dtype=dtype)
    path = tmp_path / "small.ats"
    _write_ats(path, raw, 123.456)
    result, casts = _copy_site_calls(
        lambda: TimeSeries.read(path, format="ats"),
        function_name="_read_timeseries_ats_file",
        filename="ats.py",
        samples=len(raw),
    )
    expected = raw.astype(np.float64) * 123.456 / 1000.0
    np.testing.assert_array_equal(
        result.value.view(np.uint64), expected.view(np.uint64)
    )
    assert result.dtype == np.dtype("float64")
    assert result.flags.owndata and result.flags.writeable
    assert str(result.unit) == "V"
    assert float(result.dt.value) == pytest.approx(1 / 256)
    assert getattr(result, "_gwexpy_io")["format"] == "ats"
    assert casts == 0


def test_ats_overflow_preserves_numpy_warning_and_raise(tmp_path):
    raw = np.array([0, 1, -1, np.iinfo(np.int64).max], dtype="<i8")
    path = tmp_path / "overflow.ats"
    _write_ats(path, raw, 1e300)
    with np.errstate(over="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = TimeSeries.read(path, format="ats")
    assert np.isinf(result.value[-1])
    assert [
        str(item.message) for item in caught if item.category is RuntimeWarning
    ] == ["overflow encountered in multiply"]
    with (
        np.errstate(over="raise"),
        pytest.raises(FloatingPointError, match="overflow encountered in multiply"),
    ):
        TimeSeries.read(path, format="ats")


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0, 1.5, -2.5, 4.0, -0.0, 8.0, 12.0, 16.0]),
        np.array(
            [
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).max,
                0,
                1,
                -1,
                2**53 + 1,
                -(2**53 + 1),
                8,
            ],
            dtype=np.int64,
        ),
    ],
    ids=["finite-float64", "int64-extrema"],
)
def test_netcdf_same_native_dtype_skips_roundtrip_cast(tmp_path, values):
    pytest.importorskip("netCDF4")
    path = tmp_path / "matrix.nc"
    source = TimeSeriesMatrix(
        np.tile(values, (2, 2, 1)), t0=1_000_000_000.125, dt=0.125, unit="V"
    )
    source.write(path, format="nc")
    result, casts = _copy_site_calls(
        lambda: TimeSeriesMatrix.read(path, format="nc"),
        function_name="lossless",
        filename="netcdf4_.py",
        samples=len(values),
    )
    np.testing.assert_array_equal(result.value, source.value)
    assert result.dtype == source.dtype
    assert str(result[0, 0].unit) == "V"
    assert float(result.t0.value).hex() == float(source.t0.value).hex()
    assert float(result.dt.value).hex() == float(source.dt.value).hex()
    assert casts == 0


def test_netcdf_nonfinite_keeps_existing_roundtrip_cast(tmp_path):
    pytest.importorskip("netCDF4")
    values = np.array([0.0, np.nan, np.inf, -np.inf, -0.0, 1.5, -2.5, 4.0])
    path = tmp_path / "nonfinite.nc"
    TimeSeriesMatrix(
        np.tile(values, (2, 2, 1)), t0=1_000_000_000.125, dt=0.125, unit="V"
    ).write(path, format="nc")
    result, casts = _copy_site_calls(
        lambda: TimeSeriesMatrix.read(path, format="nc"),
        function_name="lossless",
        filename="netcdf4_.py",
        samples=len(values),
    )
    np.testing.assert_array_equal(result.value, np.tile(values, (2, 2, 1)))
    assert casts == 8


def test_netcdf_object_cells_keep_existing_isnan_error(tmp_path):
    pytest.importorskip("netCDF4")
    from benchmarks.copy_audit.bx_fixtures import _write_matrix

    path = tmp_path / "object.nc"
    _write_matrix(path, "object_strings")
    with pytest.raises(TypeError, match="ufunc 'isnan' not supported"):
        TimeSeriesMatrix.read(path, format="nc")
